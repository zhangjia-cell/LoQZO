# -*- coding: utf-8 -*-
"""
trainer_alternating.py

LoQZO + QZO 交替优化训练器。

算法语义（与论文/实验设定对齐）：
1) LoQZO 阶段：固定量化 scale / zero point，只更新权重参数；
   - 继承 trainer_loqzo.OurTrainer 的低秩子空间扰动实现；
   - 二维权重使用 rank-r 子空间扰动，一维参数按 LoQZO 配置回退到全空间扰动；
   - 更新后仍可按照 QuZO 方式把权重量化回低比特表示。

2) QZO 阶段：固定权重和 zero point，只更新量化 scale；
   - 当前仓库的 qft 量化模块没有显式 zero point 参数，因此 zero point / codebook 默认保持不动；
   - QZO 只扰动并更新 TensorQuantizer.alpha，其中默认只更新 weight quantizer 的 alpha；
   - 使用对称有限差分估计方向导数：
       g_dir = [L(alpha + eps*z) - L(alpha - eps*z)] / (2*eps)
     再执行 scale <- scale - lr * g_dir * z；
   - 支持 DDC 风格的方向导数裁剪，降低异常 batch 造成的高方差。

交替策略：
- alt_a_steps：每个 cycle 中 LoQZO 的步数；
- alt_b_steps：每个 cycle 中 QZO-scale 的步数；
- alt_start：在该 step 前只跑 LoQZO warmup，之后开始交替。

注意：
- 为了让 QZO 有 scale 可更新，入口脚本必须使用 tuning_type=qft；
- run_alternating.py 会自动把 qft_alpha_only=False、qft_freeze_alpha=True，
  从而保证 LoQZO 阶段只看到权重参数，QZO 阶段由本文件手动更新 alpha。
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from trainer_loqzo import OurTrainer as LoQZOTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OurTrainer(LoQZOTrainer):
    """交替优化训练器：A-steps 使用 LoQZO，B-steps 使用 QZO-scale。"""

    # ========================================================
    # 交替调度
    # ========================================================
    def _use_loqzo_this_step(self) -> bool:
        """判断当前 step 是否走 LoQZO；返回 False 时走 QZO-scale。"""
        alt_start = max(0, int(getattr(self.args, "alt_start", 0)))
        global_step = int(getattr(self.state, "global_step", 0))
        if global_step < alt_start:
            return True

        a_steps = max(0, int(getattr(self.args, "alt_a_steps", 1)))
        b_steps = max(0, int(getattr(self.args, "alt_b_steps", 1)))

        # 若用户把某一阶段步数设为 0，则退化为单阶段训练。
        if a_steps <= 0 and b_steps <= 0:
            return True
        if b_steps <= 0:
            return True
        if a_steps <= 0:
            return False

        cycle = a_steps + b_steps
        step_in_cycle = global_step % cycle
        return step_in_cycle < a_steps

    # ========================================================
    # QZO-scale 参数收集与超参数解析
    # ========================================================
    def _qzo_eps(self) -> float:
        """QZO scale 扰动半径；默认复用 zo_eps，也可单独设置 qzo_eps。"""
        eps = float(getattr(self.args, "qzo_eps", 0.0) or 0.0)
        if eps <= 0:
            eps = float(getattr(self.args, "zo_eps", 1e-3))
        return eps

    def _qzo_scale_min(self) -> float:
        """scale 的最小值，避免 alpha 被更新成非正数。"""
        return float(getattr(self.args, "qzo_scale_min", 1e-8))

    def _qzo_scale_max(self) -> float:
        """scale 的固定最大值；<=0 表示不使用固定上界。"""
        return float(getattr(self.args, "qzo_scale_max", 0.0) or 0.0)

    def _qzo_scale_max_mult(self) -> float:
        """相对于初始 scale absmax 的最大倍率；<=0 表示关闭相对上界。"""
        return float(getattr(self.args, "qzo_scale_max_mult", 10.0) or 0.0)

    def _qzo_upper_bound_for_param(self, name: str, param: torch.nn.Parameter) -> Optional[float]:
        """计算某个 alpha 参数的上界。优先使用固定 qzo_scale_max，否则使用初始值倍率。"""
        fixed_max = self._qzo_scale_max()
        if fixed_max > 0:
            return fixed_max

        max_mult = self._qzo_scale_max_mult()
        if max_mult <= 0:
            return None

        if not hasattr(self, "_qzo_initial_scale_absmax"):
            self._qzo_initial_scale_absmax = {}
        if name not in self._qzo_initial_scale_absmax:
            init_absmax = float(param.data.detach().abs().max().item())
            init_absmax = max(init_absmax, self._qzo_scale_min())
            self._qzo_initial_scale_absmax[name] = init_absmax
        return self._qzo_initial_scale_absmax[name] * max_mult

    def _qzo_clamp_scale_(self, name: str, param: torch.nn.Parameter) -> None:
        """对 alpha 做下界/上界裁剪，防止 scale 非正或异常爆炸。"""
        lower = self._qzo_scale_min()
        upper = self._qzo_upper_bound_for_param(name, param)
        if upper is None:
            param.data.clamp_(min=lower)
        else:
            param.data.clamp_(min=lower, max=upper)

    def _qzo_scale_lr(self) -> float:
        """QZO scale 学习率；默认等于主学习率，可用 qzo_scale_lr_mult 放大/缩小。"""
        mult = float(getattr(self.args, "qzo_scale_lr_mult", 1.0))
        return float(self._get_learning_rate()) * mult

    def _qzo_clip_value(self, projected_grad):
        """DDC 风格方向导数裁剪。"""
        if not bool(getattr(self.args, "clip_zo_grad", False)):
            return projected_grad
        threshold = float(getattr(self.args, "qzo_clip_threshold", 100.0))
        if threshold <= 0:
            return projected_grad
        if isinstance(projected_grad, torch.Tensor):
            return projected_grad.clamp(min=-threshold, max=threshold)
        return max(-threshold, min(threshold, float(projected_grad)))

    def _qzo_scale_scope(self) -> str:
        """
        scale 更新范围：
        - weight：只更新权重量化 scale，即 *.quant_weight.alpha（默认，最贴近 QZO）
        - activation：只更新激活量化 scale，即 *.quant_input.alpha
        - all：同时更新权重和激活 scale（做消融时可用）
        """
        scope = str(getattr(self.args, "qzo_scale_scope", "weight")).lower().strip()
        if scope not in {"weight", "activation", "all"}:
            logger.warning("未知 qzo_scale_scope=%s，回退为 weight", scope)
            scope = "weight"
        return scope

    def _qzo_name_in_scope(self, name: str) -> bool:
        if not name.endswith("alpha"):
            return False
        scope = self._qzo_scale_scope()
        if scope == "weight":
            return ".quant_weight.alpha" in name
        if scope == "activation":
            return ".quant_input.alpha" in name
        return (".quant_weight.alpha" in name) or (".quant_input.alpha" in name)

    def _qzo_collect_scale_parameters(self, model) -> List[Tuple[str, torch.nn.Parameter]]:
        """
        收集 QZO 阶段需要扰动/更新的 scale 参数。

        注意：这些 alpha 在 LoQZO 阶段通常 requires_grad=False，
        但 QZO 是手动零阶更新，所以这里不能用 requires_grad 过滤。
        """
        # QZO 的 alpha 参数列表在 qft 包装后基本固定。
        # 缓存后可以避免每个 B-step 都遍历一次 7B 模型的全部 named_parameters。
        if bool(getattr(self.args, "qzo_cache_scale_params", True)) and hasattr(self, "_qzo_scale_parameters_cache"):
            return self._qzo_scale_parameters_cache

        scales: List[Tuple[str, torch.nn.Parameter]] = []
        for name, param in model.named_parameters():
            if self._qzo_name_in_scope(name):
                scales.append((name, param))

        if bool(getattr(self.args, "qzo_cache_scale_params", True)):
            self._qzo_scale_parameters_cache = scales

        if len(scales) == 0:
            # 中文说明：FP_W32A32 / W32A32 这类全精度前向不会创建 quant_weight.alpha。
            # 这种配置下 QZO-scale 没有可优化的 scale，因此自动退化为 LoQZO 权重更新，
            # 避免为了跑表里的 FP W32A32 变体而直接报错。
            if not getattr(self, "_qzo_no_scale_warned", False):
                logger.warning(
                    "当前模型没有 QZO 可更新的 scale 参数；若是 FP_W32A32，这是正常现象，"
                    "B 阶段会自动退化为 LoQZO 权重更新。"
                )
                self._qzo_no_scale_warned = True
            return scales

        if not getattr(self, "_qzo_scale_count_logged", False):
            logger.info(
                "QZO-scale 已收集 %d 个 alpha 参数，scope=%s，eps=%s，scale_lr_mult=%s",
                len(scales),
                self._qzo_scale_scope(),
                self._qzo_eps(),
                getattr(self.args, "qzo_scale_lr_mult", 1.0),
            )
            self._qzo_scale_count_logged = True
        return scales

    # ========================================================
    # QZO-scale 扰动生成
    # ========================================================
    def _qzo_sample_scale_delta(
        self,
        param: torch.nn.Parameter,
        seed: int,
        which: str = "q1",
    ) -> torch.Tensor:
        """
        生成 scale 的随机方向 z。

        qzo_layerwise_scale_perturb=True 时，一个 alpha 张量共享一个标量扰动，
        方差更低但自由度更小；False 时每个 scale 元素独立扰动。
        """
        torch.manual_seed(seed)
        dtype = param.data.dtype
        device = param.data.device

        if bool(getattr(self.args, "qzo_layerwise_scale_perturb", False)):
            z = torch.normal(mean=0.0, std=1.0, size=(1,), device=device, dtype=dtype)
            return z.expand_as(param.data)

        return torch.normal(
            mean=0.0,
            std=1.0,
            size=param.data.size(),
            device=device,
            dtype=dtype,
        )

    def _qzo_get_scale_deltas(self, random_seed: int) -> List[torch.Tensor]:
        """返回当前 seed 对应的 QZO scale 扰动方向。

        同一个 QZO step 的 +eps、-eps 和 update 必须使用同一个随机方向。
        原实现会重复生成 3 次相同方向；缓存后算法不变，但减少随机数生成和 Python 循环开销。
        """
        use_cache = bool(getattr(self.args, "qzo_cache_scale_deltas", True))
        cache_key = int(random_seed)
        if use_cache and hasattr(self, "_qzo_scale_delta_cache") and cache_key in self._qzo_scale_delta_cache:
            return self._qzo_scale_delta_cache[cache_key]

        deltas: List[torch.Tensor] = []
        seed = int(random_seed)
        for _, param in self.qzo_scale_parameters:
            deltas.append(self._qzo_sample_scale_delta(param, seed, which="qzo"))
            seed += 2

        if use_cache:
            if not hasattr(self, "_qzo_scale_delta_cache"):
                self._qzo_scale_delta_cache = {}
            # 每个 step 只需要当前极少数 seed，避免无界增长。
            self._qzo_scale_delta_cache.clear()
            self._qzo_scale_delta_cache[cache_key] = deltas
        return deltas

    def _qzo_clear_scale_delta_cache(self) -> None:
        if hasattr(self, "_qzo_scale_delta_cache"):
            self._qzo_scale_delta_cache.clear()

    def _qzo_backup_scales(self) -> List[torch.Tensor]:
        """备份 scale，保证 +eps / -eps 前向之后能精确恢复。"""
        return [param.data.detach().clone() for _, param in self.qzo_scale_parameters]

    def _qzo_restore_scales(self, backups: List[torch.Tensor]) -> None:
        """恢复 scale 到备份值。"""
        for (_, param), value in zip(self.qzo_scale_parameters, backups):
            param.data.copy_(value)

    def _qzo_apply_scale_perturbation(
        self,
        random_seed: int,
        scaling_factor: float,
        base_values: Optional[List[torch.Tensor]] = None,
        which: str = "q1",
    ) -> None:
        """
        在 scale 上施加 alpha <- base_alpha + scaling_factor * eps * z。

        这里使用 base_values 重新出发，而不是先 +eps 再 -2eps，
        这样可以避免 scale clamp 后无法精确恢复的问题。
        """
        eps = self._qzo_eps()
        deltas = self._qzo_get_scale_deltas(int(random_seed))

        for idx, (name, param) in enumerate(self.qzo_scale_parameters):
            if base_values is not None:
                param.data.copy_(base_values[idx])
            param.data.add_(deltas[idx], alpha=float(scaling_factor) * eps)
            self._qzo_clamp_scale_(name, param)

    # ========================================================
    # QZO-scale step / update
    # ========================================================
    def qzo_step(self, model, inputs):
        """QZO 阶段：只扰动量化 scale，估计方向导数。"""
        inputs = self._zo_prepare_inputs_once(inputs)
        self.qzo_scale_parameters = self._qzo_collect_scale_parameters(model)
        if len(self.qzo_scale_parameters) == 0:
            # FP_W32A32 没有 alpha；此时把 B 阶段当作 LoQZO 权重更新处理。
            self._last_qzo_step_fallback_to_loqzo = True
            return super().lowbit_zo_step(model, inputs)
        self._last_qzo_step_fallback_to_loqzo = False

        iterations = int(getattr(self.args, "num_pertub", 1))
        iterations = max(1, iterations)
        projected_grad_list = []
        qzo_random_seed_list = []
        loss1 = None

        for _ in range(iterations):
            qzo_seed = self._broadcast_seed()
            qzo_random_seed_list.append(qzo_seed)

            base_values = self._qzo_backup_scales()

            # L(alpha + eps * z)
            self._qzo_apply_scale_perturbation(qzo_seed, scaling_factor=1.0, base_values=base_values, which="q1")
            loss1 = self.zo_forward(model, inputs)

            # L(alpha - eps * z)
            self._qzo_apply_scale_perturbation(qzo_seed, scaling_factor=-1.0, base_values=base_values, which="q1")
            loss2 = self.zo_forward(model, inputs)

            projected_grad = self._all_reduce_mean_scalar((loss1 - loss2) / (2.0 * self._qzo_eps()))
            projected_grad = self._qzo_clip_value(projected_grad)
            projected_grad_list.append(projected_grad)

            # 恢复 alpha，真正的更新在 qzo_update 中完成。
            self._qzo_restore_scales(base_values)

        assert self.args.gradient_accumulation_steps == 1
        self.qzo_random_seed = qzo_random_seed_list
        self.qzo_projected_grad = projected_grad_list
        return loss1

    def qzo_update(self, model):
        """QZO 阶段：固定权重，只更新 scale。"""
        if bool(getattr(self, "_last_qzo_step_fallback_to_loqzo", False)):
            self._last_qzo_step_fallback_to_loqzo = False
            return super().lowbit_zo_update(model)

        iterations = len(getattr(self, "qzo_random_seed", []))
        if iterations == 0:
            raise RuntimeError("qzo_update 在 qzo_step 之前被调用，未找到随机种子。")

        lr = self._qzo_scale_lr()
        for i in range(iterations):
            qzo_seed = int(self.qzo_random_seed[i])
            projected_grad = self._qzo_clip_value(self.qzo_projected_grad[i])
            if isinstance(projected_grad, torch.Tensor):
                pg_value = float(projected_grad.detach().float().item())
            else:
                pg_value = float(projected_grad)

            deltas = self._qzo_get_scale_deltas(qzo_seed)
            for idx, (name, param) in enumerate(self.qzo_scale_parameters):
                param.data.add_(deltas[idx], alpha=-lr * pg_value)
                self._qzo_clamp_scale_(name, param)

        self._qzo_clear_scale_delta_cache()
        self.lr_scheduler.step()

    # ========================================================
    # 覆写 step / update：在每个 step 动态分派
    # ========================================================
    def lowbit_zo_step(self, model, inputs):
        """交替分派：A-steps 走 LoQZO，B-steps 走 QZO-scale。"""
        if self._use_loqzo_this_step():
            logger.debug("LoQZO step @ global_step=%d", int(getattr(self.state, "global_step", 0)))
            return super().lowbit_zo_step(model, inputs)

        logger.debug("QZO-scale step @ global_step=%d", int(getattr(self.state, "global_step", 0)))
        return self.qzo_step(model, inputs)

    def lowbit_zo_update(self, model):
        """交替分派：A-steps 走 LoQZO，B-steps 走 QZO-scale。"""
        if self._use_loqzo_this_step():
            logger.debug("LoQZO update @ global_step=%d", int(getattr(self.state, "global_step", 0)))
            return super().lowbit_zo_update(model)

        logger.debug("QZO-scale update @ global_step=%d", int(getattr(self.state, "global_step", 0)))
        return self.qzo_update(model)

    def lowbit_zo_ftstep(self, model, inputs):
        """兼容 zo_lowbit_ft：交替逻辑与主 lowbit_zo_step 相同。"""
        return self.lowbit_zo_step(model, inputs)

    def lowbit_zo_ftupdate(self, model):
        """兼容 zo_lowbit_ft：交替逻辑与主 lowbit_zo_update 相同。"""
        return self.lowbit_zo_update(model)
