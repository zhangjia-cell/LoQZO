
# -*- coding: utf-8 -*-
"""
trainer_loqzo.py

LoQZO 原型训练器：
- 继承当前仓库里的 trainer_new.OurTrainer
- 当 args.loqzo_enable=False 时，行为回退到原始 QuZO / MeZO / FO
- 当 args.loqzo_enable=True 且 trainer 被规范成 zo_lowbit / zo_lowbit_ft 时，
  用“LoZO 低秩矩阵扰动”替换原来的全空间扰动

当前实现定位：
1) 这是一个可运行的 LoQZO-Fixed / AdaRank-LoQZO 原型；
2) 支持单卡、单进程多卡模型并行、以及 DDP 的标量同步；
3) 自适应部分支持“自适应 rank”，并支持 LoZO 风格的低秩扰动采样；
   低秩扰动写成 U @ V.T：U 在每个 LoQZO step 由当前 ZO seed 重新采样，
   V 作为 lazy 子空间基每隔 loqzo_v_update_freq 个 LoQZO step 重新采样；
4) 对二维参数（线性层权重 / LoRA A/B 等）使用低秩子空间；
   对一维参数（bias / norm）默认回退到全空间 QuZO 方向，保持兼容性。
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from trainer_new import (
    OurTrainer as BaseTrainer,
    zo_quant,
    zo_dequant,
    zo_quant_data,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OurTrainer(BaseTrainer):

    def _compact_logs(self, logs: Dict) -> Dict:
        compact = dict(logs)
        epoch_progress = compact.pop("epoch", None)
        if epoch_progress is not None:
            total_epochs = max(1, int(math.ceil(getattr(self.state, "num_train_epochs", 0) or 0)))
            epoch_idx = min(total_epochs, max(1, int(math.floor(float(epoch_progress))) + 1))
            compact["epoch_idx"] = epoch_idx
            compact["epoch_total"] = total_epochs
            compact["epoch_progress"] = round(float(epoch_progress), 4)
        for key, value in list(compact.items()):
            if isinstance(value, float):
                compact[key] = round(value, 6)
        return compact

    def log(self, logs: Dict[str, float]) -> None:
        super().log(self._compact_logs(logs))

    # ================================
    # 分布式辅助函数（若基类没有，就在这里兜底）
    # ================================
    def _dist_is_initialized(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _dist_world_size(self) -> int:
        return dist.get_world_size() if self._dist_is_initialized() else 1

    def _dist_rank(self) -> int:
        return dist.get_rank() if self._dist_is_initialized() else 0

    def _dist_device(self) -> torch.device:
        if torch.cuda.is_available():
            local_rank = getattr(self.args, "local_rank", -1)
            if local_rank not in (-1, None):
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda")
        return torch.device("cpu")

    def _broadcast_seed(self, seed: Optional[int] = None) -> int:
        if self._dist_world_size() == 1:
            return int(seed if seed is not None else np.random.randint(1000000000))
        if self._dist_rank() == 0:
            value = int(seed if seed is not None else np.random.randint(1000000000))
        else:
            value = 0
        seed_tensor = torch.tensor([value], device=self._dist_device(), dtype=torch.long)
        dist.broadcast(seed_tensor, src=0)
        return int(seed_tensor.item())

    def _all_reduce_mean_scalar(self, value):
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().reshape(1).to(self._dist_device())
        else:
            tensor = torch.tensor([float(value)], device=self._dist_device(), dtype=torch.float32)

        if self._dist_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor /= self._dist_world_size()
        return tensor[0]

    # ================================
    # LoQZO 开关与参数解析
    # ================================
    def _use_loqzo(self) -> bool:
        return bool(getattr(self.args, "loqzo_enable", False)) and int(getattr(self.args, "loqzo_rank", 0)) > 0

    def _loqzo_cache_trainable_params(self) -> bool:
        """是否缓存待优化参数列表。训练开始后 requires_grad 基本固定，缓存可避免每步遍历大模型。"""
        return bool(getattr(self.args, "cache_trainable_params", True))

    def _loqzo_fast_addmm(self) -> bool:
        """是否用 addmm_ 原地应用低秩扰动，减少完整 delta 矩阵分配。"""
        return bool(getattr(self.args, "loqzo_fast_addmm", True))

    def _loqzo_skip_non_subspace_2d(self) -> bool:
        """是否跳过没有进入低秩子空间的二维参数。

        旧实现会把 embedding / lm_head 等被排除在子空间外的大矩阵退化成全空间 ZO，
        这既不符合“在 rank-r 子空间扰动权重”的算法设定，也会显著拖慢 LLaMA-2。
        默认跳过；如需完全复现旧行为，可设 loqzo_skip_non_subspace_2d=False。
        """
        return bool(getattr(self.args, "loqzo_skip_non_subspace_2d", True))

    def _loqzo_cache_lowrank_coeff(self) -> bool:
        """是否缓存当前 step 的低秩左因子 U，避免 +eps/-eps/update 反复采样同一个 U。"""
        return bool(getattr(self.args, "loqzo_cache_lowrank_coeff", True))

    def _loqzo_fuse_restore_update(self) -> bool:
        """是否尝试融合 LoQZO 的 restore 和 update 两次 addmm_。"""
        return bool(getattr(self.args, "loqzo_fuse_restore_update", True))

    def _loqzo_update_basis(self) -> bool:
        """是否启用 LoZO lazy sampling 中的 V 周期重采样。

        中文说明：
        - False：V 初始化后保持固定，但 U 仍然在每个 LoQZO step 重新采样；
        - True ：V 每隔 loqzo_v_update_freq 个 LoQZO step 重新采样一次，
          U 始终每个 LoQZO step 重新采样，对应 LoZO 的 lazy sampling。
        """
        return bool(getattr(self.args, "loqzo_update_basis", False))

    def _loqzo_v_update_freq(self) -> int:
        """V 子空间基的刷新周期。默认 1000；兼容旧参数 loqzo_u_update_freq。"""
        freq = int(getattr(self.args, "loqzo_v_update_freq", 0) or 0)
        if freq <= 0:
            freq = int(getattr(self.args, "loqzo_u_update_freq", 1000))
        return max(1, freq)

    def _loqzo_u_update_freq(self) -> int:
        """兼容旧脚本命名：在当前 LoZO 采样中，该值等价于 V 的刷新周期。"""
        return self._loqzo_v_update_freq()

    def _loqzo_update_v_every_step(self) -> bool:
        """历史兼容开关。当前实现按 loqzo_v_update_freq 进行 lazy sampling，不再每步更新 V。"""
        return bool(getattr(self.args, "loqzo_update_v_every_step", True))

    def _loqzo_u_refresh_mode(self) -> str:
        """历史兼容接口。当前实现中 U 每步从标准正态采样，不再使用该刷新模式。"""
        return str(getattr(self.args, "loqzo_u_refresh_mode", "") or "").lower().strip()

    def _loqzo_can_fuse_restore_update(self) -> bool:
        """判断当前配置下是否可以把 restore 和 update 合并。

        loss2 后参数位于 theta - eps*(U1@V.T)。更新目标为
        theta - lr*g*(U2@V.T/r)。二者共享同一个 lazy V，因此可以把左因子
        eps*U1 - lr*g*U2/r 合成为一次 addmm_。若启用 weight_decay，则保持旧的两步逻辑。
        """
        if not self._loqzo_fuse_restore_update():
            return False
        if not self._loqzo_fast_addmm():
            return False
        if float(getattr(self.args, "weight_decay", 0.0) or 0.0) != 0.0:
            return False
        return True

    def _loqzo_should_optimize_parameter(self, name: str, param: torch.nn.Parameter) -> bool:
        """LoQZO 阶段是否真的扰动/更新该参数。"""
        if not param.requires_grad:
            return False
        if self._loqzo_should_use_subspace(name, param):
            return True
        if param.ndim == 1:
            return bool(getattr(self.args, "loqzo_fullspace_for_1d", True))
        if param.ndim == 2 and self._loqzo_skip_non_subspace_2d():
            return False
        # 兼容旧行为：非子空间二维参数可以退化成全空间扰动，但默认关闭。
        return param.ndim <= 2

    def _loqzo_get_named_parameters_to_optim(self, model) -> List[Tuple[str, torch.nn.Parameter]]:
        """返回 LoQZO 阶段需要优化的参数列表。

        原始实现每个 ZO step 都遍历一次 model.named_parameters()，且会把 lm_head / embedding
        等非子空间大矩阵退化成全空间扰动。这里默认只保留真正会被 LoQZO 扰动/更新的参数。
        """
        if self._loqzo_cache_trainable_params() and hasattr(self, "_loqzo_named_parameters_to_optim_cache"):
            return self._loqzo_named_parameters_to_optim_cache

        trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        params = [(name, param) for name, param in trainable if self._loqzo_should_optimize_parameter(name, param)]
        if self._loqzo_cache_trainable_params():
            self._loqzo_named_parameters_to_optim_cache = params
        if not getattr(self, "_loqzo_trainable_cache_logged", False):
            skipped = len(trainable) - len(params)
            logger.info(
                "LoQZO 待优化参数列表已构建：%d 个张量；跳过 %d 个 trainable 张量；cache_trainable_params=%s；skip_non_subspace_2d=%s",
                len(params),
                skipped,
                self._loqzo_cache_trainable_params(),
                self._loqzo_skip_non_subspace_2d(),
            )
            self._loqzo_trainable_cache_logged = True
        return params

    def _loqzo_coeff_bits(self) -> int:
        coeff_bits = int(getattr(self.args, "loqzo_coeff_bits", 0))
        if coeff_bits <= 0:
            coeff_bits = int(getattr(self.args, "perturb_bits", 4))
        return coeff_bits

    def _loqzo_target_tokens(self) -> Optional[List[str]]:
        raw = getattr(self.args, "loqzo_target_modules", None)
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            vals = [str(x).strip() for x in raw if str(x).strip()]
            return vals or None
        text = str(raw).strip()
        if text == "" or text.lower() == "none":
            return None
        return [x.strip() for x in text.split(",") if x.strip()]

    def _loqzo_is_embedding_like(self, name: str) -> bool:
        name_l = name.lower()
        keywords = [
            "embed_tokens",
            "word_embeddings",
            "wte",
            "lm_head",
            "embed_out",
            "shared",
        ]
        return any(k in name_l for k in keywords)

    def _loqzo_should_use_subspace(self, name: str, param: torch.nn.Parameter) -> bool:
        if not self._use_loqzo():
            return False
        if not param.requires_grad:
            return False
        if param.ndim != 2:
            return False
        if not getattr(self.args, "loqzo_include_embeddings", False) and self._loqzo_is_embedding_like(name):
            return False

        targets = self._loqzo_target_tokens()
        if targets is not None and not any(tok in name for tok in targets):
            return False

        return True

    # ================================
    # LoQZO 子空间基管理
    # ================================
    def _loqzo_default_rank_for_param(self, param: torch.nn.Parameter) -> int:
        m, n = param.shape
        base_rank = int(getattr(self.args, "loqzo_rank", 8))
        return max(1, min(base_rank, m, n))

    def _loqzo_random_normal(self, rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """按 LoZO 设定采样标准正态矩阵。"""
        if cols <= 0:
            return torch.empty(rows, 0, device=device, dtype=dtype)
        return torch.randn(rows, cols, device=device, dtype=dtype)

    def _loqzo_random_orth(self, rows: int, cols: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if cols <= 0:
            return torch.empty(rows, 0, device=device, dtype=dtype)
        # 用 float32 做 QR 更稳，再 cast 回参数 dtype
        mat = torch.randn(rows, cols, device=device, dtype=torch.float32)
        q, _ = torch.linalg.qr(mat, mode="reduced")
        return q[:, :cols].to(dtype=dtype)

    def _loqzo_weight_svd_basis(self, weight: torch.Tensor, rank: int):
        w = weight.detach().float()
        q = min(rank, min(w.shape))
        try:
            # randomized 低秩 SVD，比 full SVD 更适合大矩阵
            U, S, V = torch.svd_lowrank(w, q=q, niter=1)
            return U.to(dtype=weight.dtype), V.to(dtype=weight.dtype)
        except Exception as exc:
            logger.warning(f"SVD 初始化失败，回退到随机正态因子。原因: {exc}")
            return (
                self._loqzo_random_normal(w.shape[0], q, w.device, weight.dtype),
                self._loqzo_random_normal(w.shape[1], q, w.device, weight.dtype),
            )

    def _loqzo_make_v_basis(self, name: str, param: torch.nn.Parameter, rank: int):
        """初始化 / 重采样 LoZO lazy 子空间中的右因子 V。"""
        init_mode = str(getattr(self.args, "loqzo_basis_init", "random_normal")).lower()
        rank = max(1, min(rank, param.shape[0], param.shape[1]))
        if init_mode == "svd_weight":
            _, V = self._loqzo_weight_svd_basis(param.data, rank)
        elif init_mode == "random_orth":
            # 兼容旧实验；真正的 LOZO 默认使用 random_normal。
            V = self._loqzo_random_orth(param.shape[1], rank, param.device, param.dtype)
        else:
            V = self._loqzo_random_normal(param.shape[1], rank, param.device, param.dtype)
        return V.contiguous()

    def _ensure_loqzo_state(self):
        if not hasattr(self, "_loqzo_state"):
            self._loqzo_state: Dict[str, Dict] = {}

        created = 0
        for name, param in self.named_parameters_to_optim:
            if not self._loqzo_should_use_subspace(name, param):
                continue

            desired_rank = self._loqzo_default_rank_for_param(param)
            state = self._loqzo_state.get(name)

            if (
                state is None
                or state.get("m") != param.shape[0]
                or state.get("n") != param.shape[1]
            ):
                V = self._loqzo_make_v_basis(name, param, desired_rank)
                self._loqzo_state[name] = {
                    "V": V.to(device=param.device, dtype=param.dtype),
                    "rank": desired_rank,
                    "base_rank": desired_rank,
                    "m": param.shape[0],
                    "n": param.shape[1],
                    "score": 1.0,
                    # LoZO lazy sampling 状态：V 周期重采样；U 不入 state，而是每个 step 根据 seed 重采样。
                    "last_v_refresh_step": -1,
                    "v_version": 0,
                }
                created += 1
            else:
                # 训练过程中设备 / dtype 变化时，做一次对齐
                state["V"] = state["V"].to(device=param.device, dtype=param.dtype)

        if created > 0:
            logger.info(f"LoQZO 已初始化 {created} 个二维参数的 LoZO lazy-V 子空间。")

    def _loqzo_expand_basis(self, basis: torch.Tensor, new_cols: int) -> torch.Tensor:
        """自适应 rank 时扩展 V；新增列按 LoZO 设定从标准正态采样。"""
        if new_cols <= basis.shape[1]:
            return basis[:, :new_cols].contiguous()

        rows = basis.shape[0]
        device = basis.device
        dtype = basis.dtype
        candidate = torch.empty(rows, new_cols, device=device, dtype=dtype)
        if basis.numel() > 0:
            keep = min(basis.shape[1], new_cols)
            candidate[:, :keep] = basis[:, :keep]
        else:
            keep = 0
        candidate[:, keep:] = self._loqzo_random_normal(rows, new_cols - keep, device, dtype)
        return candidate.contiguous()

    def _loqzo_reallocate_ranks(self):
        if not self._use_loqzo() or not getattr(self.args, "loqzo_adaptive_rank", False):
            return
        if not hasattr(self, "_loqzo_state") or len(self._loqzo_state) == 0:
            return

        items = list(self._loqzo_state.items())
        scores = torch.tensor(
            [max(float(state.get("score", 1.0)), 1e-8) for _, state in items],
            dtype=torch.float64,
        )
        min_rank = max(1, int(getattr(self.args, "loqzo_rank_min", 2)))
        max_rank_arg = max(min_rank, int(getattr(self.args, "loqzo_rank_max", 64)))

        total_budget = int(getattr(self.args, "loqzo_rank_budget", 0))
        if total_budget <= 0:
            total_budget = sum(int(state.get("base_rank", state["rank"])) for _, state in items)

        total_budget = max(total_budget, len(items) * min_rank)

        raw = scores / scores.sum() * total_budget
        target_ranks = raw.round().long()

        for idx, (name, state) in enumerate(items):
            max_rank = min(max_rank_arg, state["m"], state["n"])
            target = int(torch.clamp(target_ranks[idx], min=min_rank, max=max_rank).item())
            current = int(state["rank"])
            if target == current:
                continue

            state["V"] = self._loqzo_expand_basis(state["V"], target)
            state["rank"] = target
            state["v_version"] = int(state.get("v_version", 0)) + 1

        if self._dist_rank() == 0:
            avg_rank = sum(int(state["rank"]) for _, state in items) / max(len(items), 1)
            logger.info(f"LoQZO 自适应 rank 更新完成，平均 rank={avg_rank:.2f}")

    # ================================
    # LoZO 风格 U/V 子空间采样
    # ================================
    def _loqzo_refresh_v_basis_(self, name: str, param: torch.nn.Parameter, step: int, seed: Optional[int] = None) -> None:
        """按周期重新采样 V；U 不在这里采样，而是在每个 ZO seed 下即时生成。

        中文说明：
        LOZO 的 lazy sampling 是：U 每个迭代步重新采样，V 每隔 ν 个迭代步重新采样一次。
        这里的 ν 对应 loqzo_v_update_freq；为了兼容旧脚本，也会读取 loqzo_u_update_freq。
        """
        state = self._loqzo_state[name]
        rank = int(state["rank"])
        if seed is not None:
            torch.manual_seed(int(seed))
        state["V"] = self._loqzo_make_v_basis(name, param, rank).to(device=param.device, dtype=param.dtype)
        state["last_v_refresh_step"] = int(step)
        state["v_version"] = int(state.get("v_version", 0)) + 1

    def _loqzo_maybe_update_bases(self, model=None) -> None:
        """在每个 LoQZO step 开始前按需 lazy 重采样 V。

        中文说明：
        - U 不再作为固定 basis 存储，而是在 _loqzo_sample_lowrank_left 中每步采样；
        - 默认 loqzo_update_basis=False 时，V 初始化后固定；
        - 启用 loqzo_update_basis=True 后，V 每 loqzo_v_update_freq 个 LoQZO step 随机采样一次；
        - 重采样发生在 +eps/-eps 前向之前，保证同一个 ZO step 内 q1/q2/update 使用同一个 V。
        """
        if not self._use_loqzo():
            return
        if not hasattr(self, "_loqzo_state"):
            return

        # 使用 LoQZO 自己的 step 计数，而不是全局训练 step。
        # 交替训练中 global_step 同时包含 QZO-scale 步；这里希望 V 固定的是 LoQZO 权重更新步数。
        step = int(getattr(self, "_loqzo_basis_step", 0))
        self._loqzo_basis_step = step + 1
        freq = self._loqzo_v_update_freq()
        refresh_v = bool(self._loqzo_update_basis()) and (step == 0 or step % freq == 0)

        refreshed_v = 0
        if refresh_v:
            base_seed = self._broadcast_seed()
        else:
            base_seed = None

        for idx, (name, param) in enumerate(self.named_parameters_to_optim):
            if not self._loqzo_should_use_subspace(name, param):
                continue
            state = self._loqzo_state.get(name)
            if state is None:
                continue

            # 保证 V 的 device/dtype 始终与当前参数一致。
            state["V"] = state["V"].to(device=param.device, dtype=param.dtype)

            if refresh_v and int(state.get("last_v_refresh_step", -1)) != step:
                self._loqzo_refresh_v_basis_(name, param, step, seed=int(base_seed) + idx * 2)
                refreshed_v += 1

        # V 改变后清空缓存，避免同一 seed 下旧 U 与新 V 混用造成语义混淆。
        if refreshed_v > 0:
            self._loqzo_clear_lowrank_coeff_cache()

        if self._dist_rank() == 0 and (refreshed_v > 0 or not getattr(self, "_loqzo_lazy_sampling_logged", False)):
            logger.info(
                "LoQZO/LOZO 子空间采样：sample_U=every_step，refresh_V=%d，step=%d，v_update_freq=%d，basis_init=%s",
                refreshed_v,
                step,
                freq,
                getattr(self.args, "loqzo_basis_init", "random_normal"),
            )
            self._loqzo_lazy_sampling_logged = True

    # ================================
    # LoQZO 扰动生成
    # ================================
    def _loqzo_sample_fullspace_delta(self, param, seed: int, which: str = "q1"):
        torch.manual_seed(seed)
        fp_perturb = torch.normal(
            mean=0,
            std=1,
            size=param.data.size(),
            device=param.data.device,
            dtype=param.data.dtype,
        )
        q_seed = seed + (11 if which == "q1" else 23)
        if bool(getattr(self.args, "quantized_perturb_ours", False)):
            quantized_perturb, s, z = zo_quant(
                fp_perturb,
                nbits=int(getattr(self.args, "perturb_bits", 4)),
                seed=q_seed,
                stochastic=True,
                sym=True,
            )
        else:
            quantized_perturb, s, z = zo_quant(
                fp_perturb,
                nbits=int(getattr(self.args, "perturb_bits", 4)),
                seed=q_seed,
                stochastic=False,
                sym=True,
            )
        return zo_dequant(quantized_perturb, s, z)

    def _loqzo_sample_lowrank_left(self, name: str, param, seed: int, which: str = "q1") -> torch.Tensor:
        """采样 LoZO 低秩扰动中的左因子 U。

        中文说明：
        - V 存在 _loqzo_state[name]["V"] 中，并按 lazy sampling 周期刷新；
        - U 不持久化，每个 LoQZO step 根据当前 seed 重新采样；
        - q1/q2 使用同一个 full-precision U 的两次独立量化版本，对应 QuZO 的双随机量化扰动。
        """
        state = self._loqzo_state[name]
        cache_key = (name, int(seed), str(which), int(state.get("v_version", 0)))
        if self._loqzo_cache_lowrank_coeff():
            cache = getattr(self, "_loqzo_lowrank_coeff_cache", None)
            if cache is not None and cache_key in cache:
                return cache[cache_key]

        rank = int(state["rank"])
        torch.manual_seed(seed)
        left = torch.normal(
            mean=0,
            std=1,
            size=(param.shape[0], rank),
            device=param.data.device,
            dtype=param.data.dtype,
        )

        if bool(getattr(self.args, "loqzo_quantize_coeff", True)):
            coeff_bits = self._loqzo_coeff_bits()
            q_seed = seed + (11 if which == "q1" else 23)
            if bool(getattr(self.args, "quantized_perturb_ours", False)):
                left_q, s, zp = zo_quant(left, nbits=coeff_bits, seed=q_seed, stochastic=True, sym=True)
            else:
                left_q, s, zp = zo_quant(left, nbits=coeff_bits, seed=q_seed, stochastic=False, sym=True)
            left = zo_dequant(left_q, s, zp)

        if self._loqzo_cache_lowrank_coeff():
            if not hasattr(self, "_loqzo_lowrank_coeff_cache"):
                self._loqzo_lowrank_coeff_cache = {}
            self._loqzo_lowrank_coeff_cache[cache_key] = left
        return left

    def _loqzo_clear_lowrank_coeff_cache(self) -> None:
        if hasattr(self, "_loqzo_lowrank_coeff_cache"):
            self._loqzo_lowrank_coeff_cache.clear()

    def _loqzo_sample_lowrank_delta(self, name: str, param, seed: int, which: str = "q1", divide_by_rank: bool = False):
        """旧接口：返回完整 delta。仅在关闭 loqzo_fast_addmm 或调试时使用。"""
        state = self._loqzo_state[name]
        V = state["V"]
        left = self._loqzo_sample_lowrank_left(name, param, seed, which=which)
        delta = left.matmul(V.transpose(0, 1))
        if divide_by_rank:
            delta = delta / max(1, int(state["rank"]))
        return delta

    def _loqzo_lowrank_delta_mean_square(self, name: str, left: torch.Tensor, divide_by_rank: bool = False) -> float:
        """估计 ||U V^T||_F^2 / (m*n)，不构造完整 m*n delta。

        对 LOZO 的随机正态 U/V，Frobenius 范数可由两个 r*r Gram 矩阵计算：
            ||U V^T||_F^2 = tr((U^T U)(V^T V))。
        update 方向按 LOZO 使用 U V^T / r，因此 divide_by_rank=True 时再除以 r^2。
        """
        state = self._loqzo_state[name]
        V = state["V"].to(device=left.device, dtype=left.dtype)
        rank = max(1, int(state["rank"]))
        denom = max(1, int(state.get("m", 1)) * int(state.get("n", 1)))
        left_f = left.detach().float()
        v_f = V.detach().float()
        gram_left = left_f.transpose(0, 1).matmul(left_f)
        gram_v = v_f.transpose(0, 1).matmul(v_f)
        norm_sq = torch.sum(gram_left * gram_v)
        if divide_by_rank:
            norm_sq = norm_sq / float(rank * rank)
        return float(norm_sq.item() / denom)

    def _loqzo_add_lowrank_left_(
        self,
        name: str,
        param,
        left: torch.Tensor,
        alpha: float = 1.0,
        divide_by_rank: bool = False,
        return_score: bool = False,
    ) -> float:
        """用给定左因子原地加上 alpha * U V^T；update 阶段可除以 rank。"""
        state = self._loqzo_state[name]
        V = state["V"]
        rank_scale = 1.0 / max(1, int(state["rank"])) if divide_by_rank else 1.0
        param.data.addmm_(left, V.transpose(0, 1), beta=1.0, alpha=float(alpha) * rank_scale)
        if return_score:
            return self._loqzo_lowrank_delta_mean_square(name, left, divide_by_rank=divide_by_rank)
        return 0.0

    def _loqzo_add_lowrank_(
        self,
        name: str,
        param,
        seed: int,
        alpha: float,
        which: str = "q1",
        divide_by_rank: bool = False,
        return_score: bool = False,
    ) -> float:
        """原地加上 alpha * U V^T，并按需返回该扰动的 mean-square。

        这是主要加速点：不分配完整 delta，而是用 addmm_ 直接写入参数。
        LoZO 中 finite-difference 扰动使用 U V^T，参数更新使用 U V^T / r。
        """
        left = self._loqzo_sample_lowrank_left(name, param, seed, which=which)
        return self._loqzo_add_lowrank_left_(
            name,
            param,
            left,
            alpha=alpha,
            divide_by_rank=divide_by_rank,
            return_score=return_score,
        )

    def _loqzo_sample_delta(self, name: str, param, seed: int, which: str = "q1", divide_by_rank: bool = False):
        if self._loqzo_should_use_subspace(name, param):
            return self._loqzo_sample_lowrank_delta(name, param, seed, which=which, divide_by_rank=divide_by_rank)

        if param.ndim == 1 and not bool(getattr(self.args, "loqzo_fullspace_for_1d", True)):
            return None
        if param.ndim == 2 and self._loqzo_skip_non_subspace_2d():
            return None

        return self._loqzo_sample_fullspace_delta(param, seed, which=which)

    def _loqzo_perturb_parameters(self, scaling_factor=1.0, which: str = "q1"):
        seed = int(self.zo_random_seed)
        eps = float(self.args.zo_eps)
        for name, param in self.named_parameters_to_optim:
            if self._loqzo_fast_addmm() and self._loqzo_should_use_subspace(name, param):
                self._loqzo_add_lowrank_(
                    name,
                    param,
                    seed,
                    alpha=float(scaling_factor) * eps,
                    which=which,
                    divide_by_rank=False,
                    return_score=False,
                )
            else:
                delta = self._loqzo_sample_delta(name, param, seed, which=which, divide_by_rank=False)
                if delta is not None:
                    param.data.add_(delta, alpha=float(scaling_factor) * eps)
            seed += 2

    def _loqzo_update_score_from_mean_square(self, name: str, delta_mean_square: float, projected_grad):
        if name not in getattr(self, "_loqzo_state", {}):
            return
        state = self._loqzo_state[name]
        beta = float(getattr(self.args, "loqzo_rank_ema", 0.9))
        if isinstance(projected_grad, torch.Tensor):
            pg_abs = float(projected_grad.detach().float().abs().item())
        else:
            pg_abs = abs(float(projected_grad))
        score_inc = pg_abs * float(delta_mean_square)
        state["score"] = beta * float(state.get("score", 1.0)) + (1.0 - beta) * score_inc

    def _loqzo_update_score(self, name: str, delta: torch.Tensor, projected_grad):
        self._loqzo_update_score_from_mean_square(name, float(delta.float().pow(2).mean().item()), projected_grad)

    # ================================
    # 覆盖：LoQZO 版本的 step / update
    # ================================
    def lowbit_zo_step(self, model, inputs):
        if not self._use_loqzo():
            return super().lowbit_zo_step(model, inputs)

        args = self.args
        inputs = self._zo_prepare_inputs_once(inputs)

        # 当前要优化的参数列表。默认缓存，避免每个 step 都遍历一次 7B 模型。
        self.named_parameters_to_optim = self._loqzo_get_named_parameters_to_optim(model)

        self._ensure_loqzo_state()
        # LoZO 风格子空间采样：在本 step 采样扰动之前按 lazy 规则更新 V。
        # U 会在当前 ZO seed 下每步重新采样；同一个 step 内 q1/q2/update 共享一致的 V。
        self._loqzo_maybe_update_bases(model)

        iterations = int(getattr(self.args, "num_pertub", 1))
        projected_grad_list = []
        zo_random_seed_list = []
        can_fuse_restore_update = self._loqzo_can_fuse_restore_update() and iterations == 1
        self._loqzo_pending_fused_restore = False

        for _ in range(iterations):
            self.zo_random_seed = self._broadcast_seed()
            zo_random_seed_list.append(self.zo_random_seed)

            # +eps
            self._loqzo_perturb_parameters(scaling_factor=1.0, which="q1")
            loss1 = self.zo_forward(model, inputs)

            # -eps
            self._loqzo_perturb_parameters(scaling_factor=-2.0, which="q1")
            loss2 = self.zo_forward(model, inputs)

            projected_grad = self._all_reduce_mean_scalar((loss1 - loss2) / (2 * self.args.zo_eps))

            # 标量也做一次轻量量化，和当前 QuZO 代码习惯保持一致
            max_abs_value = torch.max(torch.abs(projected_grad))
            # 避免 float(s) 触发 GPU 同步；零值时用极小下界兜底。
            s = torch.clamp(max_abs_value / 127, min=1e-12)
            quantized_grad = torch.round(projected_grad / s)
            quantized_grad = torch.clamp(quantized_grad, -127, 127)
            projected_grad = quantized_grad * s

            projected_grad_list.append(projected_grad)

            # 恢复原参数。若可以融合 restore+update，则这里先不恢复，
            # 让 update 阶段用一次 addmm_ 完成 theta-eps*(U1V.T) -> theta-lr*g*(U2V.T/r)。
            if can_fuse_restore_update:
                self._loqzo_pending_fused_restore = True
            else:
                self._loqzo_perturb_parameters(scaling_factor=1.0, which="q1")

        self.zo_random_seed = zo_random_seed_list
        self.projected_grad = projected_grad_list
        return loss1

    def lowbit_zo_update(self, model):
        if not self._use_loqzo():
            return super().lowbit_zo_update(model)

        args = self.args
        iterations = int(getattr(self.args, "num_pertub", 1))

        adaptive_rank_enabled = bool(getattr(self.args, "loqzo_adaptive_rank", False))

        for i in range(iterations):
            zo_random_seed_update = int(self.zo_random_seed[i])
            projected_grad = self.projected_grad[i]
            lr = float(self._get_learning_rate())
            # projected_grad 是一个标量。addmm_ 的 alpha 需要 Python number，
            # 因此每个 perturb iteration 只同步一次，避免在每层循环中反复 .item()。
            if isinstance(projected_grad, torch.Tensor):
                pg_value = float(projected_grad.detach().float().item())
            else:
                pg_value = float(projected_grad)

            seed = zo_random_seed_update
            if bool(getattr(self, "_loqzo_pending_fused_restore", False)) and self._loqzo_can_fuse_restore_update():
                # 当前参数仍在 theta - eps*(U1V.T)。目标更新为 theta - lr*pg*(U2V.T/r)。
                # 二者共享同一个 lazy V，因此一次 addmm_ 写入 [eps*U1-lr*pg*U2/r]V.T 即可。
                eps = float(self.args.zo_eps)
                for name, param in self.named_parameters_to_optim:
                    if self._loqzo_fast_addmm() and self._loqzo_should_use_subspace(name, param):
                        state = self._loqzo_state[name]
                        rank = max(1, int(state["rank"]))
                        left1 = self._loqzo_sample_lowrank_left(name, param, seed, which="q1")
                        left2 = self._loqzo_sample_lowrank_left(name, param, seed, which="q2")
                        left_combined = eps * left1 - (lr * pg_value / rank) * left2
                        self._loqzo_add_lowrank_left_(name, param, left_combined, alpha=1.0, divide_by_rank=False, return_score=False)
                        if adaptive_rank_enabled:
                            delta_ms = self._loqzo_lowrank_delta_mean_square(name, left2, divide_by_rank=True)
                            self._loqzo_update_score_from_mean_square(name, delta_ms, projected_grad)
                    else:
                        # 默认配置不会走到这里；为了安全，回退到用 q1 方向恢复并用 q2/r 更新。
                        delta1 = self._loqzo_sample_delta(name, param, seed, which="q1", divide_by_rank=False)
                        delta2 = self._loqzo_sample_delta(name, param, seed, which="q2", divide_by_rank=True)
                        if delta1 is not None:
                            param.data.add_(delta1, alpha=eps)
                        if delta2 is not None:
                            param.data.add_(delta2, alpha=-lr * pg_value)

                    # 与普通分支保持一致：INT 权重更新后重新压回整数低比特表示。
                    qbits = int(getattr(self.args, "wbit", 16))
                    wmode = str(getattr(self.args, "wmode", getattr(self.args, "mode", "int"))).lower()
                    if qbits < 16 and wmode == "int":
                        stochastic = bool(getattr(self.args, "quantized_perturb_ours", False))
                        qd, scaling, zero = zo_quant_data(param.data, nbits=qbits, stochastic=stochastic, sym=True)
                        param.data = zo_dequant(qd, scaling, zero)
                    seed += 2

                self._loqzo_pending_fused_restore = False
                continue

            for name, param in self.named_parameters_to_optim:
                is_norm_or_bias = ("bias" in name or "layer_norm" in name or "layernorm" in name)

                if self._loqzo_fast_addmm() and self._loqzo_should_use_subspace(name, param):
                    # 对二维权重：theta <- theta * (1-lr*wd) - lr*pg*(U V^T / r)。
                    # 用 addmm_ 原地完成低秩更新，不构造完整 delta。
                    if not is_norm_or_bias and float(args.weight_decay) != 0.0:
                        param.data.mul_(1.0 - lr * float(args.weight_decay))
                    delta_ms = self._loqzo_add_lowrank_(
                        name,
                        param,
                        seed,
                        alpha=-lr * pg_value,
                        which="q2",
                        divide_by_rank=True,
                        return_score=adaptive_rank_enabled,
                    )
                    if adaptive_rank_enabled:
                        self._loqzo_update_score_from_mean_square(name, delta_ms, pg_value)
                else:
                    # 对一维参数或关闭 fast_addmm 时，保留旧逻辑。
                    pg = projected_grad.to(device=param.data.device, dtype=param.data.dtype) if isinstance(projected_grad, torch.Tensor) else projected_grad
                    delta = self._loqzo_sample_delta(name, param, seed, which="q2", divide_by_rank=True)
                    if delta is None:
                        seed += 2
                        continue

                    if not is_norm_or_bias:
                        if float(args.weight_decay) != 0.0:
                            param.data.mul_(1.0 - lr * float(args.weight_decay))
                        param.data.add_(delta, alpha=-lr * float(pg_value))
                    else:
                        param.data.add_(delta, alpha=-lr * float(pg_value))

                    if adaptive_rank_enabled:
                        self._loqzo_update_score(name, delta, projected_grad)

                # 更新后是否把潜在权重重新压回整数低比特表示。
                # 中文说明：
                #   - INT_W8A8 / INTFP_W4A8：wmode=int，权重本身按整数低比特保存/更新；
                #   - FP_W8A8：wmode=float，权重 latent 保持浮点，只在前向中通过 float codebook 量化；
                #   - FP_W32A32：wbit>=16，不做权重量化。
                # 这样可以区分 QuZO 表里的 FP、INT、INT/FP，而不是只看 WBIT/ABIT。
                qbits = int(getattr(self.args, "wbit", 16))
                wmode = str(getattr(self.args, "wmode", getattr(self.args, "mode", "int"))).lower()
                if qbits < 16 and wmode == "int":
                    stochastic = bool(getattr(self.args, "quantized_perturb_ours", False))
                    qd, scaling, zero = zo_quant_data(param.data, nbits=qbits, stochastic=stochastic, sym=True)
                    param.data = zo_dequant(qd, scaling, zero)

                seed += 2

        # 自适应 rank：只依赖历史 score，不依赖当前步重新估计的同一组随机扰动
        if bool(getattr(self.args, "loqzo_adaptive_rank", False)):
            freq = max(1, int(getattr(self.args, "loqzo_rank_update_freq", 200)))
            if (int(getattr(self.state, "global_step", 0)) + 1) % freq == 0:
                self._loqzo_reallocate_ranks()

        self._loqzo_pending_fused_restore = False
        self._loqzo_clear_lowrank_coeff_cache()
        self.lr_scheduler.step()

    def lowbit_zo_ftstep(self, model, inputs):
        # 先给一个简单兼容：LoQZO-FT 复用 LoQZO 主 step
        return self.lowbit_zo_step(model, inputs)

    def lowbit_zo_ftupdate(self, model):
        return self.lowbit_zo_update(model)
