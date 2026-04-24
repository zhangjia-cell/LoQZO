import logging

# 配置日志输出格式，便于在注入 prefix、冻结参数时观察运行信息
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import torch
from torch import nn


def find_module(root_module: nn.Module, key: str):
    """
    在一个嵌套的 Transformer 模型里，按照 "a.b.c" 这样的层级名字，
    找到目标子模块，并返回：
        1) 父模块 parent_module
        2) 目标子模块在父模块里的属性名
        3) 目标子模块本身 module

    这个工具函数的用途：
    PrefixTuning 需要把每一层 attention 的 forward 替换掉，
    所以必须先靠模块名把它找出来。

    例如：
        key = "model.layers.0.self_attn"
    那么最后会返回：
        parent_module = model.layers.0
        sub_name      = "self_attn"
        module        = model.layers.0.self_attn
    """
    sub_keys = key.split(".")
    parent_module = root_module

    # 逐层往下走，直到目标模块的上一层
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)

    # 取出最后一级，也就是目标模块
    module = getattr(parent_module, sub_keys[-1])

    return parent_module, sub_keys[-1], module


def attn_forward_hook(self, *args, **kwargs):
    """
    这是“替换原 attention.forward”之后真正会执行的新 forward。

    它做的事情可以概括成一句话：
        在 attention 的 K/V cache 前面，插入可训练的 prefix key/value。

    Prefix Tuning 的本质就是：
        不改主模型大部分参数，
        只额外学习一小段“前缀表示”，
        然后把它当成 attention 的额外上下文。

    在 decoder-only 模型里，这通常通过 past_key_value 注入。
    """

    def _expand_bsz(x, bsz):
        """
        把 prefix 从 [num_prefix, hidden]
        变成 attention 需要的 batch 形式：
            [bsz, num_heads, num_prefix, head_dim]

        原来：
            x.shape = (num_prefix, hidden_dim)

        先 reshape 成：
            (num_prefix, num_heads, head_dim)

        再 transpose 成：
            (num_heads, num_prefix, head_dim)

        最后在 batch 维复制：
            (bsz, num_heads, num_prefix, head_dim)
        """
        x = x.reshape(x.size(0), self.num_heads, -1).transpose(0, 1)
        x = x.unsqueeze(0).expand(bsz, *x.shape)
        return x

    # 兼容两种调用方式：
    # 1) hidden_states 作为关键字参数传进来
    # 2) hidden_states 作为第一个位置参数传进来
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states = args[0]

    # 当前 batch size
    bsz = hidden_states.size(0)

    # 只在“当前还没有 past_key_value”的时候注入 prefix
    # 这通常对应生成的第一步，或普通前向时首次进入该层
    if 'past_key_value' not in kwargs or kwargs['past_key_value'] is None:

        # 如果使用 reparameterization：
        # prefix 不是直接学 K/V，而是先学 prefix_input_embeds，
        # 再通过 MLP 映射成 keys / values
        if self.reparam:
            prefix_keys = self.prefix_mlp_keys(self.prefix_input_embeds)
            prefix_values = self.prefix_mlp_values(self.prefix_input_embeds)
        else:
            # 不用重参数化时，直接学习 prefix_keys / prefix_values
            prefix_keys, prefix_values = self.prefix_keys, self.prefix_values

        # 把 prefix K/V 塞到 past_key_value 里
        kwargs['past_key_value'] = (
            _expand_bsz(prefix_keys, bsz),
            _expand_bsz(prefix_values, bsz)
        )

        # attention_mask 也必须同步扩展：
        # 因为现在序列前面多了 num_prefix 个“虚拟前缀 token”
        #
        # 这里前面拼接的是 0（通过 -torch.zeros 得到，数值还是 0），
        # 在 attention mask 语义里表示：
        # prefix 这些位置是可见的，不应该被 mask 掉。
        if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
            am = kwargs['attention_mask']
            kwargs['attention_mask'] = torch.cat(
                [
                    -torch.zeros((*am.shape[:-1], self.num_prefix), dtype=am.dtype, device=am.device),
                    am
                ],
                dim=-1
            )
        elif len(args) > 1:
            # 有些模型把 attention_mask 作为位置参数传入
            am = args[1]
            am = torch.cat(
                [
                    -torch.zeros((*am.shape[:-1], self.num_prefix), dtype=am.dtype, device=am.device),
                    am
                ],
                dim=-1
            )
            args = (args[0], am) + args[2:]

    # 最后仍然调用原始 attention.forward，
    # 只是我们在调用前悄悄改好了 past_key_value 和 attention_mask
    return self.original_forward(*args, **kwargs)


def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    """
    替换模型原来的 prepare_inputs_for_generation，
    让 HuggingFace 的 generate() 在 prefix tuning 场景下也能正常工作。

    为什么要改？
    因为 generate 过程中，模型会不断复用 past_key_values。
    一旦前面额外拼了 prefix，attention_mask 的长度也必须和它匹配，
    否则 shape 会对不上。
    """

    # 这个变量在当前代码里其实没有被真正使用到，可以删掉也不影响逻辑
    original_input_len = input_ids.size(-1)

    # 如果已经有 past_key_values，说明不是生成第一步，
    # 只需要把最后一个 token 喂进去即可
    if past_key_values:
        input_ids = input_ids[:, -1:]

    # HuggingFace generate 的标准逻辑：
    # 如果第一次调用时给了 inputs_embeds，就只在第一步用它
    if inputs_embeds is not None and past_key_values is None:
        model_inputs = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    # 如果 past_key_values 存在，需要检查 attention_mask 长度是否已经包含 prefix
    if past_key_values is not None:
        # past_key_values[0][0].size(2) = cached key 的 seq_len
        # attention_mask.size(1) - 1      = 当前非 prefix 的历史长度
        #
        # 若两者不等，说明 cached KV 里多出来的部分就是 prefix
        if past_key_values[0][0].size(2) != attention_mask.size(1) - 1:
            num_prefix = past_key_values[0][0].size(2) - (attention_mask.size(1) - 1)

            # 给 attention_mask 前面补 1，表示 prefix 位置可见
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (attention_mask.size(0), num_prefix),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    ),
                    attention_mask
                ],
                dim=-1
            )

    # 按 HF generate 约定返回 model_inputs
    model_inputs.update(
        {
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


class PrefixTuning:
    """
    这是整个文件的核心类。

    它不是定义一个“新模型”，而是对现有模型做“就地注入”：
        1) 找到所有 attention 层
        2) 替换 attention.forward
        3) 给每层 attention 增加 prefix 参数
        4) 冻结非 prefix 参数
        5) 修补 generate() 相关输入准备函数

    所以它更像一个“模型改造器 / 注入器”。
    """

    def __init__(
        self,
        model,
        num_prefix,
        reparam=True,
        embed_dim=512,
        mid_dim=512,
        float16=False,
        init_by_real_act=False
    ):
        """
        参数说明：
        - model:
            要注入 prefix tuning 的原始模型
        - num_prefix:
            prefix token 的个数
        - reparam:
            是否使用重参数化技巧
            True: 学 prefix_input_embeds，再经 MLP 生成 K/V
            False: 直接学习 prefix_keys / prefix_values
        - embed_dim, mid_dim:
            重参数化 MLP 的维度
        - float16:
            是否把 prefix 的 MLP 也转成 fp16
        - init_by_real_act:
            是否用真实 token 的激活来初始化 prefix（而不是纯随机）
        """

        self.model = model
        self.num_prefix = num_prefix
        self.hidden_dim = model.config.hidden_size
        self.float16 = float16

        # 重参数化相关配置
        self.reparam = reparam
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim

        # 这个变量用于“跨层共享第一层的 prefix_input_embeds”
        # 只在 reparam=True 时用到
        input_embeds = None

        # 不同模型的 attention 模块命名不同，所以先做模型类型分流
        if model.config.model_type == "opt":
            attention_name = "attn"
            first_layer_name = "layers.0"
            layer_name = "layers."
        elif model.config.model_type == "roberta":
            attention_name = "attention"
            first_layer_name = "layer.0"
            layer_name = "layer."
        elif model.config.model_type == "llama":
            attention_name = "self_attn"
            first_layer_name = "layers.0"
            layer_name = "layers."
        else:
            raise NotImplementedError

        # 用真实激活初始化 prefix
        # 这个分支要求不能使用 reparam
        if init_by_real_act:
            assert not reparam

            # 随机采样一些 token id，长度等于 num_prefix
            input_tokens = torch.randint(
                low=0,
                high=model.config.vocab_size,
                size=(1, num_prefix),
                dtype=torch.long
            ).cuda()

            # 通过模型前向，取出每层真实的 past_key_values 作为初始化来源
            if model.config.model_type in ["opt", "llama"]:
                with torch.no_grad():
                    real_key_values = model(input_ids=input_tokens, use_cache=True).past_key_values
            else:
                raise NotImplementedError

        # 遍历模型中所有子模块，把 attention 层都替换掉
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                # 取出层号，仅用于日志和从 real_key_values 中索引
                layer_id = int(key.split(layer_name)[1].split(".")[0])
                logger.info(f"Inject prefix to: {key}")

                # 找到这一层的 attention 模块
                _, _, attn = find_module(model, key)

                # 备份原始 forward，再用我们的 hook 覆盖
                attn.original_forward = attn.forward
                attn.forward = attn_forward_hook.__get__(attn, type(attn))

                # 有些 attention 模块可能没有显式 num_heads 属性，手动补上
                if not hasattr(attn, "num_heads"):
                    attn.num_heads = model.config.num_attention_heads

                # 判断是不是第一层
                first = first_layer_name in key

                # 给这一层 attention 注入 prefix 参数
                self.add_prefix(attn, first=first, input_embeds=input_embeds)

                # 如果使用重参数化，只在第一层真正创建 prefix_input_embeds，
                # 后续层共享同一个 prefix_input_embeds
                if first and self.reparam:
                    input_embeds = attn.prefix_input_embeds

                # 如果要求用真实激活初始化，就把这一层 prefix_keys / prefix_values
                # 替换成真实前向得到的 KV
                if init_by_real_act:
                    logger.info(f"Reinitialize with actual activation: {key} (layer {layer_id})")
                    keys = real_key_values[layer_id][0].squeeze(0).transpose(0, 1).reshape(num_prefix, -1)
                    values = real_key_values[layer_id][1].squeeze(0).transpose(0, 1).reshape(num_prefix, -1)
                    attn.prefix_keys.data = keys.to(attn.prefix_keys.data.device)
                    attn.prefix_values.data = values.to(attn.prefix_values.data.device)

        # 冻结所有非 prefix 参数
        # 这就是 prefix tuning 的参数高效所在：
        # 主模型不动，只训练 prefix 相关参数
        for n, p in model.named_parameters():
            if "prefix" not in n:
                p.requires_grad = False

        # 替换掉模型的 generation 输入准备逻辑，保证 generate() 兼容 prefix
        model.prepare_inputs_for_generation = prepare_inputs_for_generation.__get__(model, type(model))

    def add_prefix(self, module, first, input_embeds=None):
        """
        给某一层 attention 模块真正挂上 prefix 参数。

        参数：
        - module:
            某一个 attention 层
        - first:
            是否为第一层
        - input_embeds:
            reparam 模式下，后续层复用第一层的 prefix_input_embeds
        """
        # 默认把 prefix 参数创建在 k_proj 所在设备上，避免 device mismatch
        device = module.k_proj.weight.data.device

        # 把一些属性直接挂到 attention 模块上，供 attn_forward_hook 读取
        module.num_prefix = self.num_prefix
        module.reparam = self.reparam

        if self.reparam:
            # ========== 重参数化版本 ==========
            # 不直接学 K/V，而是先学一个低维 prefix embedding，
            # 再通过 MLP 投影到 hidden_dim

            if first:
                # 第一层负责真正创建可学习的 prefix_input_embeds
                logger.info("For prefix+reparameterization, inject the embeddings in the first layer.")
                module.prefix_input_embeds = nn.Parameter(
                    torch.randn(
                        self.num_prefix,
                        self.embed_dim,
                        device=device,
                        dtype=self.model.dtype
                    ),
                    requires_grad=True
                )
            else:
                # 后续层共享第一层的 prefix_input_embeds
                assert input_embeds is not None
                module.prefix_input_embeds = input_embeds

            # 两个独立的 MLP：
            # 一个把 prefix embedding 映射成 key
            # 一个把 prefix embedding 映射成 value
            module.prefix_mlp_keys = nn.Sequential(
                nn.Linear(self.embed_dim, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.hidden_dim)
            ).to(device)

            module.prefix_mlp_values = nn.Sequential(
                nn.Linear(self.embed_dim, self.mid_dim),
                nn.Tanh(),
                nn.Linear(self.mid_dim, self.hidden_dim)
            ).to(device)

            # 如果主训练是 fp16，这里也把 prefix MLP 转成 half
            if self.float16:
                module.prefix_mlp_keys = module.prefix_mlp_keys.half()
                module.prefix_mlp_values = module.prefix_mlp_values.half()

        else:
            # ========== 非重参数化版本 ==========
            # 直接学习 prefix key / value
            module.prefix_keys = nn.Parameter(
                torch.randn(
                    self.num_prefix,
                    self.hidden_dim,
                    device=device,
                    dtype=self.model.dtype
                ),
                requires_grad=True
            )
            module.prefix_values = nn.Parameter(
                torch.randn(
                    self.num_prefix,
                    self.hidden_dim,
                    device=device,
                    dtype=self.model.dtype
                ),
                requires_grad=True
            )
