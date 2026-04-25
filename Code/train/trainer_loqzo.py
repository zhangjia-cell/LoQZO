
# -*- coding: utf-8 -*-
"""
trainer_loqzo.py

LoQZO 原型训练器：
- 继承当前仓库里的 trainer_new.OurTrainer
- 当 args.loqzo_enable=False 时，行为回退到原始 QuZO / MeZO / FO
- 当 args.loqzo_enable=True 且 trainer 被规范成 zo_lowbit / zo_lowbit_ft 时，
  用“低秩子空间系数扰动”替换原来的全空间扰动

当前实现定位：
1) 这是一个可运行的 LoQZO-Fixed / AdaRank-LoQZO 原型；
2) 支持单卡、单进程多卡模型并行、以及 DDP 的标量同步；
3) 自适应部分当前先实现“自适应 rank”，不在这里做复杂的 basis 旋转；
4) 对二维参数（线性层权重 / LoRA A/B 等）使用低秩子空间；
   对一维参数（bias / norm）默认回退到全空间 QuZO 方向，保持兼容性。
"""

import logging
import math
from typing import Dict, List, Optional

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
            logger.warning(f"SVD 初始化失败，回退到随机正交基。原因: {exc}")
            return (
                self._loqzo_random_orth(w.shape[0], q, w.device, weight.dtype),
                self._loqzo_random_orth(w.shape[1], q, w.device, weight.dtype),
            )

    def _loqzo_make_basis(self, name: str, param: torch.nn.Parameter, rank: int):
        init_mode = str(getattr(self.args, "loqzo_basis_init", "random_orth")).lower()
        rank = max(1, min(rank, param.shape[0], param.shape[1]))
        if init_mode == "svd_weight":
            U, V = self._loqzo_weight_svd_basis(param.data, rank)
        else:
            U = self._loqzo_random_orth(param.shape[0], rank, param.device, param.dtype)
            V = self._loqzo_random_orth(param.shape[1], rank, param.device, param.dtype)
        return U, V

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
                U, V = self._loqzo_make_basis(name, param, desired_rank)
                self._loqzo_state[name] = {
                    "U": U.to(device=param.device, dtype=param.dtype),
                    "V": V.to(device=param.device, dtype=param.dtype),
                    "rank": desired_rank,
                    "base_rank": desired_rank,
                    "m": param.shape[0],
                    "n": param.shape[1],
                    "score": 1.0,
                }
                created += 1
            else:
                # 训练过程中设备 / dtype 变化时，做一次对齐
                state["U"] = state["U"].to(device=param.device, dtype=param.dtype)
                state["V"] = state["V"].to(device=param.device, dtype=param.dtype)

        if created > 0:
            logger.info(f"LoQZO 已初始化 {created} 个二维参数的低秩子空间。")

    def _loqzo_expand_basis(self, basis: torch.Tensor, new_cols: int) -> torch.Tensor:
        if new_cols <= basis.shape[1]:
            return basis[:, :new_cols].contiguous()

        rows = basis.shape[0]
        device = basis.device
        dtype = basis.dtype
        candidate = torch.randn(rows, new_cols, device=device, dtype=torch.float32)
        if basis.numel() > 0:
            keep = min(basis.shape[1], new_cols)
            candidate[:, :keep] = basis[:, :keep].float()
        q, _ = torch.linalg.qr(candidate, mode="reduced")
        return q[:, :new_cols].to(dtype=dtype)

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

            state["U"] = self._loqzo_expand_basis(state["U"], target)
            state["V"] = self._loqzo_expand_basis(state["V"], target)
            state["rank"] = target

        if self._dist_rank() == 0:
            avg_rank = sum(int(state["rank"]) for _, state in items) / max(len(items), 1)
            logger.info(f"LoQZO 自适应 rank 更新完成，平均 rank={avg_rank:.2f}")

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

    def _loqzo_sample_lowrank_delta(self, name: str, param, seed: int, which: str = "q1"):
        state = self._loqzo_state[name]
        rank = int(state["rank"])
        U = state["U"]
        V = state["V"]

        torch.manual_seed(seed)
        z = torch.normal(
            mean=0,
            std=1,
            size=(rank,),
            device=param.data.device,
            dtype=param.data.dtype,
        )

        if bool(getattr(self.args, "loqzo_quantize_coeff", True)):
            coeff_bits = self._loqzo_coeff_bits()
            q_seed = seed + (11 if which == "q1" else 23)
            if bool(getattr(self.args, "quantized_perturb_ours", False)):
                z_q, s, zp = zo_quant(z, nbits=coeff_bits, seed=q_seed, stochastic=True, sym=True)
            else:
                z_q, s, zp = zo_quant(z, nbits=coeff_bits, seed=q_seed, stochastic=False, sym=True)
            z_hat = zo_dequant(z_q, s, zp)
        else:
            z_hat = z

        # U diag(z) V^T 采用更省显存的实现： (U * z[None, :]) @ V^T
        delta = (U * z_hat.unsqueeze(0)) @ V.transpose(0, 1)
        return delta

    def _loqzo_sample_delta(self, name: str, param, seed: int, which: str = "q1"):
        if self._loqzo_should_use_subspace(name, param):
            return self._loqzo_sample_lowrank_delta(name, param, seed, which=which)

        if param.ndim == 1 and not bool(getattr(self.args, "loqzo_fullspace_for_1d", True)):
            return None

        return self._loqzo_sample_fullspace_delta(param, seed, which=which)

    def _loqzo_perturb_parameters(self, scaling_factor=1.0, which: str = "q1"):
        seed = int(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            delta = self._loqzo_sample_delta(name, param, seed, which=which)
            if delta is not None:
                param.data = param.data + scaling_factor * delta * self.args.zo_eps
            seed += 2

    def _loqzo_update_score(self, name: str, delta: torch.Tensor, projected_grad):
        if name not in getattr(self, "_loqzo_state", {}):
            return
        state = self._loqzo_state[name]
        beta = float(getattr(self.args, "loqzo_rank_ema", 0.9))
        score_inc = float(abs(float(projected_grad))) * float(delta.float().pow(2).mean().item())
        state["score"] = beta * float(state.get("score", 1.0)) + (1.0 - beta) * score_inc

    # ================================
    # 覆盖：LoQZO 版本的 step / update
    # ================================
    def lowbit_zo_step(self, model, inputs):
        if not self._use_loqzo():
            return super().lowbit_zo_step(model, inputs)

        args = self.args

        # 当前要优化的参数列表
        self.named_parameters_to_optim = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        self._ensure_loqzo_state()

        iterations = int(getattr(self.args, "num_pertub", 1))
        projected_grad_list = []
        zo_random_seed_list = []

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
            s = max_abs_value / 127
            if float(s) == 0.0:
                s = torch.tensor(1e-3, device=projected_grad.device)
            quantized_grad = torch.round(projected_grad / s)
            quantized_grad = torch.clamp(quantized_grad, -127, 127)
            projected_grad = quantized_grad * s

            projected_grad_list.append(projected_grad)

            # 恢复原参数
            self._loqzo_perturb_parameters(scaling_factor=1.0, which="q1")

        self.zo_random_seed = zo_random_seed_list
        self.projected_grad = projected_grad_list
        return loss1

    def lowbit_zo_update(self, model):
        if not self._use_loqzo():
            return super().lowbit_zo_update(model)

        args = self.args
        iterations = int(getattr(self.args, "num_pertub", 1))

        for i in range(iterations):
            zo_random_seed_update = int(self.zo_random_seed[i])
            projected_grad = self.projected_grad[i]

            seed = zo_random_seed_update
            for name, param in self.named_parameters_to_optim:
                # model_parallel 时参数可能分布在不同 GPU，上一步得到的 projected_grad
                # 需要移动到当前参数所在设备，避免 cuda:0 与 cuda:1 相乘报错。
                pg = projected_grad.to(device=param.data.device, dtype=param.data.dtype) if isinstance(projected_grad, torch.Tensor) else projected_grad
                delta = self._loqzo_sample_delta(name, param, seed, which="q2")
                if delta is None:
                    seed += 2
                    continue

                if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                    param.data = param.data - self._get_learning_rate() * (
                        pg * delta + args.weight_decay * param.data
                    )
                else:
                    param.data = param.data - self._get_learning_rate() * pg * delta

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

        self.lr_scheduler.step()

    def lowbit_zo_ftstep(self, model, inputs):
        # 先给一个简单兼容：LoQZO-FT 复用 LoQZO 主 step
        return self.lowbit_zo_step(model, inputs)

    def lowbit_zo_ftupdate(self, model):
        return self.lowbit_zo_update(model)
