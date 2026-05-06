#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# alternating_loqzo_qzo.sh
# ------------------------------------------------------------
# LoQZO + QZO(scale) 交替优化训练脚本。
#
# 算法含义：
#   A 阶段（LoQZO）：固定 scale / zero point，只在 rank-r 子空间内扰动并更新权重；
#   B 阶段（QZO）：固定权重 / zero point，只扰动并更新量化 scale(alpha)。
#
# 这个脚本只用于“我们自己的交替算法”及其消融：
#   ALT_A_STEPS=1 ALT_B_STEPS=1  -> LoQZO + QZO-scale 交替优化
#   ALT_A_STEPS=1 ALT_B_STEPS=0  -> 只跑 LoQZO 权重低秩更新
#   ALT_A_STEPS=0 ALT_B_STEPS=1  -> 只跑 QZO-scale 更新
#
# 重要要求：
#   1) 必须使用 TYPE=qft / MODE=qft，因为 QZO-scale 阶段需要模型里有 quant_weight.alpha；
#   2) run_alternating.py 会自动设置 qft_freeze_alpha=True、qft_alpha_only=False，
#      保证 LoQZO 阶段不更新 scale，QZO 阶段再手动更新 scale；
#   3) WBIT/ABIT 控制 qft 量化模块，PBIT 控制 QuZO/LoQZO 扰动量化位宽。
# ============================================================

# ============================================================
# 命令行轻量解析
# ------------------------------------------------------------
# 支持在 bash 后面额外传一个日志尾缀：
#   bash Code/script/alternating_loqzo_qzo.sh --logname ablation_r8
#   bash Code/script/alternating_loqzo_qzo.sh --logname=ablation_r8
# 也支持用环境变量：
#   logname=ablation_r8 bash Code/script/alternating_loqzo_qzo.sh
#   LOGNAME_SUFFIX=ablation_r8 bash Code/script/alternating_loqzo_qzo.sh
#
# 注意：不建议使用大写 LOGNAME，因为 Linux 通常已经用 LOGNAME 表示当前用户名。
# 其它未知参数会继续透传给 Python 入口。
# ============================================================
CLI_LOGNAME_SUFFIX=""
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --logname)
      if [[ $# -lt 2 ]]; then
        echo "错误：--logname 需要跟一个非空尾缀，例如 --logname exp1" >&2
        exit 1
      fi
      CLI_LOGNAME_SUFFIX="$2"
      shift 2
      ;;
    --logname=*)
      CLI_LOGNAME_SUFFIX="${1#*=}"
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${PASSTHROUGH_ARGS[@]}"

# -------------------- 定位项目路径 --------------------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PROJECT_ROOT=$(cd "$CODE_ROOT/.." && pwd)
cd "$PROJECT_ROOT"

# -------------------- CUDA / quant_cuda 环境 --------------------
export CUDA_HOME="${CUDA_HOME:-/home/zhangjia/cuda-12.1}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
append_ld_path_if_exists() {
  local p="$1"
  if [ -n "$p" ] && [ -d "$p" ]; then
    case ":$LD_LIBRARY_PATH:" in *":$p:"*) ;; *) export LD_LIBRARY_PATH="$p${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;; esac
  fi
}
append_ld_path_if_exists "$CUDA_HOME/lib64"
append_ld_path_if_exists "$CUDA_HOME/targets/x86_64-linux/lib"

# quant_cuda 扩展所在目录。
# 如果服务器 Python 版本不是 3.11，请先在 Code/quant 下重新编译：
#   cd Code/quant && python setup.py build_ext --inplace
PY_TAG=$(python - <<'PY'
import sys
print(f"{sys.version_info.major}{sys.version_info.minor}")
PY
)
QUANT_LIB_DIR="$CODE_ROOT/quant/build/lib.linux-x86_64-cpython-${PY_TAG}"
PYTHONPATH_ENTRIES=("$CODE_ROOT" "$CODE_ROOT/train" "$CODE_ROOT/quant")
if [ -d "$QUANT_LIB_DIR" ]; then
  PYTHONPATH_ENTRIES+=("$QUANT_LIB_DIR")
else
  while IFS= read -r d; do PYTHONPATH_ENTRIES+=("$d"); done < <(find "$CODE_ROOT/quant/build" -maxdepth 1 -type d -name 'lib.*' 2>/dev/null || true)
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_ENTRIES[*]}")${PYTHONPATH:+:$PYTHONPATH}"
unset BNB_CUDA_VERSION

# ============================================================
# 模型参数 MODEL / MODEL_PATH
# ------------------------------------------------------------
# MODEL 可以填 HuggingFace repo id、本地模型目录名、或 run_loqzo.py 中注册过的别名。
# 常用可选项如下：
#
#   OPT 系列：
#     MODEL=facebook/opt-125m
#     MODEL=facebook/opt-1.3b
#     MODEL=facebook/opt-2.7b
#     MODEL=facebook/opt-6.7b
#     MODEL=facebook/opt-13b
#     MODEL=facebook/opt-30b
#     MODEL=OPT-125M       # run_loqzo.py 内置别名
#     MODEL=OPT-1.3B       # run_loqzo.py 内置别名
#     MODEL=OPT-2.7B       # run_loqzo.py 内置别名
#     MODEL=OPT-13B        # run_loqzo.py 内置别名
#     MODEL=OPT-30B        # run_loqzo.py 内置别名
#
#   LLaMA / Mistral 系列：
#     MODEL=meta-llama/Llama-2-7b-hf
#     MODEL=meta-llama/Llama-2-13b-hf
#     MODEL=meta-llama/Meta-Llama-3-8B
#     MODEL=mistralai/Mistral-7B-v0.3
#     MODEL=mistralai/Mistral-7B-Instruct-v0.3
#     MODEL=Llama2-7B      # run_loqzo.py 内置别名，需要你有对应权限/本地权重
#     MODEL=Llama2-13B     # run_loqzo.py 内置别名，需要你有对应权限/本地权重
#     MODEL=Llama3-8B      # run_loqzo.py 内置别名，需要你有对应权限/本地权重
#     MODEL=Mistral-7B     # run_loqzo.py 内置别名
#
# MODEL_PATH 优先级最高。如果你已经把模型下载到本地，推荐显式指定：
#   MODEL_PATH=/home/zhangjia/Code/LoQZO/LoQZO/Models/base_models/opt-1.3b
# 此时 MODEL 只用于命名和记录，实际加载源以 MODEL_PATH 为准。
#
# PREFER_LOCAL_MODEL 推荐保持 False，避免 MODEL=facebook/opt-1.3b 被模糊匹配到 opt-13b。
# ============================================================
MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_PATH=${MODEL_PATH:-/root/NeurIPS2026/Models/base_models/meta-llama--Llama-2-7b-hf}
PREFER_LOCAL_MODEL=${PREFER_LOCAL_MODEL:-False}

# ============================================================
# 任务参数 TASK
# ------------------------------------------------------------
# TASK 可选项分三类：
#
#   分类 / 多选任务：
#     SST2      情感二分类，表格中一般写 SST2
#     RTE       文本蕴含二分类
#     WSC       Winograd Schema Challenge，二分类
#     WIC       Word-in-Context，二分类
#     BoolQ     是/否问答，二分类
#     CB        三分类自然语言推理，小数据集，默认 DEV=0
#     MultiRC   多句阅读理解二分类
#     Copa      因果推理二选一，小数据集，默认 DEV=0
#     WinoGrande 常识推理二选一
#
#   生成 / 阅读理解任务：
#     ReCoRD    SuperGLUE ReCoRD，实体填空，多候选
#     SQuAD     生成式问答，使用 F1/EM 一类指标
#     DROP      生成式阅读理解，使用 F1/EM 一类指标
#
#   语言建模任务：
#     WikiText  语言建模 / 困惑度类任务
#
# 支持大小写别名，例如 TASK=sst2、TASK=record、TASK=squad 通常也可以。
# ============================================================
TASK=${TASK:-SST2}
RUN_PY=${RUN_PY:-$CODE_ROOT/train/run_alternating.py}
LOG_HOME=${LOG_HOME:-./outputs}
TAG=${TAG:-}
LOG_PREFIX=${LOG_PREFIX:-alt_loqzo_qzo}
TIME_TAG=${TIME_TAG:-$(date +"%Y%m%d-%H%M%S")}
LOGNAME_SUFFIX="${CLI_LOGNAME_SUFFIX:-${LOGNAME_SUFFIX:-${logname:-}}}"

MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

# -------------------- 文件名安全化函数 --------------------
sanitize_name() {
  # 把空格、斜杠、冒号等替换为下划线，避免日志文件名非法。
  echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//'
}
SAFE_LOG_SUFFIX=$(sanitize_name "$LOGNAME_SUFFIX")

# ============================================================
# GPU / 启动方式
# ------------------------------------------------------------
# GPU_ID 是唯一需要常用改动的 GPU 参数：
#
#   单卡：GPU_ID=0
#   多卡：GPU_ID=0,1            # 单进程多卡，默认 LAUNCH_MODE=auto -> model_parallel
#   多卡：GPU_ID=0,1,2,3
#
# LAUNCH_MODE 可选：
#   auto            自动选择；GPU_ID 只有一张卡时 single，多张卡时 model_parallel
#   single          单进程单卡；如果 GPU_ID=0,1 但强制 single，也只会按普通单进程启动
#   model_parallel  单进程多卡模型并行，适合 ZO/LoQZO 大模型实验
#   ddp             torchrun DDP，多进程数据并行；更适合一阶训练，不一定适合 ZO
#   zero3           torchrun + DeepSpeed ZeRO-3；显存紧张时可尝试
#
# 兼容旧写法：GPU_IDS=0,1 也会被当成 GPU_ID=0,1。
# ============================================================
GPU_ID=${GPU_ID:-${GPU_IDS:-}}
GPU_ID=${GPU_ID//[[:space:]]/}
LAUNCH_MODE=${LAUNCH_MODE:-auto}
GPU_ARGS=()
NUM_PROC=1
if [ -n "$GPU_ID" ]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_ID"
  NUM_PROC=0
  for gid in "${GPU_ID_ARRAY[@]}"; do
    [ -n "$gid" ] && NUM_PROC=$((NUM_PROC + 1))
  done
  [ "$NUM_PROC" -lt 1 ] && NUM_PROC=1
  if [ "$NUM_PROC" -eq 1 ]; then
    GPU_ARGS+=(--gpu_id "$GPU_ID")
  else
    GPU_ARGS+=(--gpu_ids "$GPU_ID")
  fi
fi
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-$NUM_PROC}
TORCHRUN_EXTRA_ARGS=${TORCHRUN_EXTRA_ARGS:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-$CODE_ROOT/script/ds_zero3_bf16.json}

# ============================================================
# 训练轮数 / 数据比例
# ------------------------------------------------------------
# 你主要调这两个：
#   STEPS=5000        按总优化步数训练；当 EPOCHS=0 时生效
#   EPOCHS=3          按 epoch 训练；当 EPOCHS>0 时会自动忽略 STEPS
#
# 数据采样数量默认按任务自动设置，不需要每次手动改 TRAIN/DEV/EVAL。
# 如果确实要覆盖，也可以显式传入：TRAIN=1000 DEV=500 EVAL=1000。
#
# DEV 的含义：从 train split 里额外切出多少样本做训练期评估。
# 小数据集如 CB/Copa/WSC 默认 DEV=0，避免把训练集切空。
# 最终测试/验证仍使用原始 validation split，并由 EVAL 控制最多采多少条。
# ============================================================
BS=${BS:-16}
GRAD_ACC=${GRAD_ACC:-1}
REAL_BS=$((BS * GRAD_ACC))
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
STEPS=${STEPS:-5000}
EPOCHS=${EPOCHS:-${TRAIN_EPOCHS:-0}}
SAVE_STEPS=${SAVE_STEPS:-50000}
LOGGING_STEPS=${LOGGING_STEPS:-20}
EVAL_STEPS=${EVAL_STEPS:-1000}
# FAST_TASK_LENGTH=True 时，如果用户没有显式传 MAX_LENGTH，脚本会按任务给更省时的默认长度。
# 需要完全复现旧设置时，显式传 FAST_TASK_LENGTH=False MAX_LENGTH=512。
MAX_LENGTH_USER_SET=0
if [[ -n "${MAX_LENGTH+x}" ]]; then
  MAX_LENGTH_USER_SET=1
fi
FAST_TASK_LENGTH=${FAST_TASK_LENGTH:-True}
MAX_LENGTH=${MAX_LENGTH:-512}
NO_EVAL=${NO_EVAL:-False}
EVAL_DURING_TRAINING=${EVAL_DURING_TRAINING:-False}
OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR:-False}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}

TASK_LOWER=$(echo "$TASK" | tr '[:upper:]' '[:lower:]')
DEFAULT_TRAIN=1000
DEFAULT_DEV=500
DEFAULT_EVAL=1000
case "$TASK_LOWER" in
  cb)
    DEFAULT_TRAIN=250; DEFAULT_DEV=0; DEFAULT_EVAL=56 ;;
  copa)
    DEFAULT_TRAIN=400; DEFAULT_DEV=0; DEFAULT_EVAL=100 ;;
  wsc|wsc.fixed)
    DEFAULT_TRAIN=554; DEFAULT_DEV=0; DEFAULT_EVAL=104 ;;
  rte)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=256; DEFAULT_EVAL=277 ;;
  wic)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=256; DEFAULT_EVAL=638 ;;
  boolq)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=500; DEFAULT_EVAL=1000 ;;
  sst2|sst-2)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=500; DEFAULT_EVAL=872 ;;
  multirc|multi_rc)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=500; DEFAULT_EVAL=1000 ;;
  record|recordd|record_task|record_dataset)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=500; DEFAULT_EVAL=1000 ;;
  squad|squad_v1|squad1)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=500; DEFAULT_EVAL=1000 ;;
  drop)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=500; DEFAULT_EVAL=1000 ;;
  winogrande)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=500; DEFAULT_EVAL=1000 ;;
  wikitext)
    DEFAULT_TRAIN=1000; DEFAULT_DEV=0; DEFAULT_EVAL=1000 ;;
esac
# 根据任务自动压缩默认输入长度。
# 说明：MultiRC/BoolQ 等阅读类分类任务在 LLaMA-2-7B 上用 512 会非常慢；
# 默认 256 能明显降低训练时长。若论文实验需要固定 512，请显式设置 MAX_LENGTH=512 或 FAST_TASK_LENGTH=False。
if [[ "$MAX_LENGTH_USER_SET" == "0" && ("$FAST_TASK_LENGTH" == "True" || "$FAST_TASK_LENGTH" == "true") ]]; then
  case "$TASK_LOWER" in
    multirc|multi_rc|boolq|rte|wic|wsc|wsc.fixed|cb|copa|sst2|sst-2|winogrande)
      MAX_LENGTH=256 ;;
    record|recordd|record_task|record_dataset|squad|squad_v1|squad1|drop)
      MAX_LENGTH=512 ;;
  esac
fi

TRAIN=${TRAIN:-$DEFAULT_TRAIN}
DEV=${DEV:-$DEFAULT_DEV}
EVAL=${EVAL:-$DEFAULT_EVAL}

# -------------------- 方法固定为交替优化 --------------------
METHOD=${METHOD:-alternating}
TRAINER=${TRAINER:-alternating}
TRAINER_MODULE=${TRAINER_MODULE:-trainer_alternating}
MODE=${MODE:-qft}
TYPE=${TYPE:-qft}
OPTIM=${OPTIM:-sgd}
NUM_PERTUB=${NUM_PERTUB:-1}
MASK_RATIO=${MASK_RATIO:-0}
TWO=${TWO:-True}

# ============================================================
# qft 量化参数：对齐 QuZO 表格的 Data Precision
# ------------------------------------------------------------
# 你现在主要调 PRECISION_PROFILE，而不是只调 WBIT/ABIT。
#
# 可选：
#   PRECISION_PROFILE=FP_W32A32
#       表格左侧：FP / W32A32
#       含义：不量化权重和激活，等价全精度前向。
#       注意：没有 quant_weight.alpha，所以 QZO-scale 阶段没有 scale 可更新；
#             脚本默认把 ALT_B_STEPS 改成 0，算法退化为 LoQZO-only 全精度权重更新。
#
#   PRECISION_PROFILE=FP_W8A8
#       表格左侧：FP / W8A8
#       含义：权重和激活都是 8-bit floating-point codebook。
#       映射：WBIT=8, ABIT=8, WMODE=float, AMODE=float。
#
#   PRECISION_PROFILE=INT_W8A8
#       表格左侧：INT / W8A8
#       含义：权重和激活都是 8-bit integer codebook。
#       映射：WBIT=8, ABIT=8, WMODE=int, AMODE=int。
#
#   PRECISION_PROFILE=INTFP_W4A8
#       表格左侧：INT/FP / W4A8
#       含义：权重用 INT4，激活用 FP8。
#       映射：WBIT=4, ABIT=8, WMODE=int, AMODE=float。
#
# 兼容写法：DATA_PRECISION 或 PRECISION 也可以作为别名。
# 例如：
#   DATA_PRECISION=INT_W8A8 bash Code/script/alternating_loqzo_qzo.sh
#   PRECISION=FP_W8A8 bash Code/script/alternating_loqzo_qzo.sh
#
# 如果你不想用 profile，也可以手动指定：
#   PRECISION_PROFILE= WBIT=4 ABIT=8 QMODE=int WMODE=int AMODE=int ...
# 这会走手动模式。
# ============================================================
PRECISION_PROFILE=${PRECISION_PROFILE-${DATA_PRECISION-${PRECISION-INTFP_W4A8}}}
PRECISION_KEY=$(echo "$PRECISION_PROFILE" | tr '[:upper:]' '[:lower:]' | sed -E 's#[/ -]+#_#g')

# 记录用户是否显式设置过 ALT_B_STEPS。FP_W32A32 默认禁用 QZO-scale，
# 但如果你显式传 ALT_B_STEPS，也不会被脚本强行覆盖；训练器里会自动 fallback。
ALT_B_STEPS_USER_SET=0
if [[ -n "${ALT_B_STEPS+x}" ]]; then
  ALT_B_STEPS_USER_SET=1
fi

case "$PRECISION_KEY" in
  fp_w32a32|fp32|w32a32)
    PRECISION_PROFILE=FP_W32A32
    WBIT=32; ABIT=32; QMODE=float; WMODE=float; AMODE=float
    ;;
  fp_w8a8|fp8|float_w8a8)
    PRECISION_PROFILE=FP_W8A8
    WBIT=8; ABIT=8; QMODE=float; WMODE=float; AMODE=float
    ;;
  int_w8a8|int8|w8a8)
    PRECISION_PROFILE=INT_W8A8
    WBIT=8; ABIT=8; QMODE=int; WMODE=int; AMODE=int
    ;;
  intfp_w4a8|int_fp_w4a8|int_fp|w4a8)
    PRECISION_PROFILE=INTFP_W4A8
    WBIT=4; ABIT=8; QMODE=mixed; WMODE=int; AMODE=float
    ;;
  ""|manual|none)
    PRECISION_PROFILE=""
    WBIT=${WBIT:-4}
    ABIT=${ABIT:-8}
    QMODE=${QMODE:-int}
    WMODE=${WMODE:-$QMODE}
    AMODE=${AMODE:-$QMODE}
    ;;
  *)
    echo "不支持的 PRECISION_PROFILE=$PRECISION_PROFILE" >&2
    echo "可选：FP_W32A32 / FP_W8A8 / INT_W8A8 / INTFP_W4A8；或 PRECISION_PROFILE=manual 后手动传 WBIT/ABIT/WMODE/AMODE。" >&2
    exit 1
    ;;
esac

PBIT=${PBIT:-4}
NO_OUTLIER=${NO_OUTLIER:-False}

# -------------------- 大模型加载精度 / 速度 --------------------
# Data Precision 仍由 PRECISION_PROFILE 控制；这里的 LOAD_BFLOAT16/LOAD_FLOAT16 只是“原始模型加载 dtype”。
# A100 上 LLaMA/Mistral/OPT-6.7B+ 用 BF16 通常比 FP32 更快、更省显存，且不等于加载外部量化模型。
LOAD_FLOAT16_USER_SET=0
LOAD_BFLOAT16_USER_SET=0
if [[ -n "${LOAD_FLOAT16+x}" ]]; then LOAD_FLOAT16_USER_SET=1; fi
if [[ -n "${LOAD_BFLOAT16+x}" ]]; then LOAD_BFLOAT16_USER_SET=1; fi
AUTO_BF16_FOR_LARGE_MODEL=${AUTO_BF16_FOR_LARGE_MODEL:-True}
LOAD_FLOAT16=${LOAD_FLOAT16:-False}
LOAD_BFLOAT16=${LOAD_BFLOAT16:-False}
MODEL_LOWER_FOR_DTYPE=$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')
if [[ "$AUTO_BF16_FOR_LARGE_MODEL" == "True" || "$AUTO_BF16_FOR_LARGE_MODEL" == "true" ]]; then
  if [[ "$LOAD_FLOAT16_USER_SET" == "0" && "$LOAD_BFLOAT16_USER_SET" == "0" ]]; then
    case "$MODEL_LOWER_FOR_DTYPE" in
      *llama*|*mistral*|*opt-6.7b*|*opt-13b*|*opt-30b*)
        LOAD_BFLOAT16=True ;;
    esac
  fi
fi
LOAD_INT8=${LOAD_INT8:-False}
LOAD_INT4=${LOAD_INT4:-False}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
FORCE_DISABLE_BNB=${FORCE_DISABLE_BNB:-False}

# -------------------- ZO/LoQZO 加速开关 --------------------
# 这些开关不改变 LoQZO/QZO 的数学更新；主要减少多余 forward、参数遍历和完整 delta 分配。
SKIP_POST_UPDATE_FORWARD=${SKIP_POST_UPDATE_FORWARD:-True}
CACHE_TRAINABLE_PARAMS=${CACHE_TRAINABLE_PARAMS:-True}
LOQZO_FAST_ADDMM=${LOQZO_FAST_ADDMM:-True}
QZO_CACHE_SCALE_PARAMS=${QZO_CACHE_SCALE_PARAMS:-True}
QZO_CACHE_SCALE_DELTAS=${QZO_CACHE_SCALE_DELTAS:-True}
# 同一个 ZO step 的 +eps/-eps 两次 forward 复用已经搬到 GPU 的 batch，避免重复 CPU->GPU 拷贝。
PREPARE_INPUTS_ONCE_FOR_ZO=${PREPARE_INPUTS_ONCE_FOR_ZO:-True}
# ZO forward 中模型一直是 eval，避免每次 forward 递归 model.eval()。
FAST_ZO_EVAL_MODE=${FAST_ZO_EVAL_MODE:-True}
# 训练 loss 不需要 past_key_values，关闭 use_cache 避免无用 KV cache。
DISABLE_TRAIN_USE_CACHE=${DISABLE_TRAIN_USE_CACHE:-True}
# qft 时只训练被量化包装后的 Linear 权重，冻结 lm_head/embedding/norm 等未量化参数，避免全空间扰动。
QFT_TRAIN_QUANTIZED_LINEAR_ONLY=${QFT_TRAIN_QUANTIZED_LINEAR_ONLY:-True}
QFT_TRAIN_BIAS=${QFT_TRAIN_BIAS:-False}
# 训练时只对 option token 位置计算 lm_head/log_softmax，避免整段 prompt 的 full-vocab logits。
# 对 only_train_option=True 的分类/多选任务有效；生成任务自动不受影响。
FAST_OPTION_LOSS=${FAST_OPTION_LOSS:-True}
# 评估时把同一个样本的多个 candidate 合成 batch，并复用 FAST_OPTION_LOSS 路径。
FAST_EVAL_BATCH_OPTIONS=${FAST_EVAL_BATCH_OPTIONS:-True}
# ZO 大模型训练时关闭每 step FLOPs 统计和 NaN/Inf Tensor bool 检查，避免 GPU 同步。
SKIP_FLOS_METER_FOR_ZO=${SKIP_FLOS_METER_FOR_ZO:-True}
DISABLE_NAN_INF_FILTER_FOR_ZO=${DISABLE_NAN_INF_FILTER_FOR_ZO:-True}
# int + no_outlier=True 时启用 round/clamp 快速量化路径；不设 NO_OUTLIER 时不会改变原通用 codebook 路径。
FAST_INT_QUANT=${FAST_INT_QUANT:-True}

# -------------------- LoQZO 低秩子空间参数 --------------------
LOQZO_ENABLE=${LOQZO_ENABLE:-True}
LOQZO_RANK=${LOQZO_RANK:-8}
LOQZO_ADAPTIVE_RANK=${LOQZO_ADAPTIVE_RANK:-False}
LOQZO_RANK_MIN=${LOQZO_RANK_MIN:-2}
LOQZO_RANK_MAX=${LOQZO_RANK_MAX:-64}
LOQZO_RANK_BUDGET=${LOQZO_RANK_BUDGET:-0}
LOQZO_RANK_UPDATE_FREQ=${LOQZO_RANK_UPDATE_FREQ:-200}
LOQZO_RANK_EMA=${LOQZO_RANK_EMA:-0.9}
LOQZO_BASIS_INIT=${LOQZO_BASIS_INIT:-random_normal}  # random_normal / random_orth / svd_weight
# LoZO 风格 lazy sampling：默认开启。
# 含义：U 每个 LoQZO step 按当前 ZO seed 重新采样；V 每 LOQZO_V_UPDATE_FREQ 个 LoQZO step 重新采样。
LOQZO_UPDATE_BASIS=${LOQZO_UPDATE_BASIS:-True}
LOQZO_V_UPDATE_FREQ=${LOQZO_V_UPDATE_FREQ:-0}
# 兼容旧环境变量名：当 LOQZO_V_UPDATE_FREQ=0 时，Python 端会读取 LOQZO_U_UPDATE_FREQ 作为 V 的刷新周期。
LOQZO_U_UPDATE_FREQ=${LOQZO_U_UPDATE_FREQ:-1000}
LOQZO_UPDATE_V_EVERY_STEP=${LOQZO_UPDATE_V_EVERY_STEP:-True}
LOQZO_U_REFRESH_MODE=${LOQZO_U_REFRESH_MODE:-}
LOQZO_TARGET_MODULES=${LOQZO_TARGET_MODULES:-}
LOQZO_INCLUDE_EMBEDDINGS=${LOQZO_INCLUDE_EMBEDDINGS:-False}
# 默认不再对 bias / norm 等一维参数做全空间扰动；你的算法重点是二维权重矩阵的 rank-r 子空间。
LOQZO_FULLSPACE_FOR_1D=${LOQZO_FULLSPACE_FOR_1D:-False}
# 默认跳过没有进入子空间的二维大矩阵，比如 lm_head / embedding，避免退化成全空间 ZO。
LOQZO_SKIP_NON_SUBSPACE_2D=${LOQZO_SKIP_NON_SUBSPACE_2D:-True}
LOQZO_CACHE_LOWRANK_COEFF=${LOQZO_CACHE_LOWRANK_COEFF:-True}
LOQZO_FUSE_RESTORE_UPDATE=${LOQZO_FUSE_RESTORE_UPDATE:-True}
LOQZO_QUANTIZE_COEFF=${LOQZO_QUANTIZE_COEFF:-True}
LOQZO_COEFF_BITS=${LOQZO_COEFF_BITS:-0}

# -------------------- 交替策略和 QZO-scale 参数 --------------------
ALT_A_STEPS=${ALT_A_STEPS:-1}                  # 每个周期 LoQZO 步数
ALT_B_STEPS=${ALT_B_STEPS:-1}                  # 每个周期 QZO-scale 步数
# FP_W32A32 不产生量化 scale(alpha)。为了避免表格中的全精度变体误跑 QZO，
# 如果用户没有显式指定 ALT_B_STEPS，则默认只跑 LoQZO 权重更新。
if [[ "$PRECISION_PROFILE" == "FP_W32A32" && "$ALT_B_STEPS_USER_SET" == "0" ]]; then
  ALT_B_STEPS=0
fi
ALT_START=${ALT_START:-0}                      # 前 ALT_START 步只跑 LoQZO warmup
QZO_EPS=${QZO_EPS:-0}                          # <=0 时复用 EPS
QZO_SCALE_LR_MULT=${QZO_SCALE_LR_MULT:-1.0}    # scale 学习率倍率
QZO_SCALE_MIN=${QZO_SCALE_MIN:-1e-8}
QZO_SCALE_MAX=${QZO_SCALE_MAX:-0}              # 固定上界；0 表示不用固定上界
QZO_SCALE_MAX_MULT=${QZO_SCALE_MAX_MULT:-10.0} # 相对初始 scale 的最大倍率；<=0 表示关闭
QZO_SCALE_SCOPE=${QZO_SCALE_SCOPE:-weight}     # weight / activation / all
QZO_LAYERWISE_SCALE_PERTURB=${QZO_LAYERWISE_SCALE_PERTURB:-False}
CLIP_ZO_GRAD=${CLIP_ZO_GRAD:-False}
QZO_CLIP_THRESHOLD=${QZO_CLIP_THRESHOLD:-100.0}

# -------------------- 最终评估设置 --------------------
# 旧脚本固定 eval_num_demos=32，会让 LLaMA2 在 MultiRC/ReCoRD/SQuAD/DROP 上最终评估非常慢。
# 主表格通常建议先用 0 demo；如需复现实验中的 few-shot evaluation，可显式设 EVAL_NUM_DEMOS=32。
USE_EVAL_DEMOS_AFTER_TRAINING=${USE_EVAL_DEMOS_AFTER_TRAINING:-True}
EVAL_NUM_DEMOS=${EVAL_NUM_DEMOS:-0}
EVAL_DEMO_SEED=${EVAL_DEMO_SEED:-0}

# -------------------- WandB / 日志 --------------------
WANDB_PROJECT=${WANDB_PROJECT:-LLM_LoQZO_github}
USE_WANDB=${USE_WANDB:-False}
REPORT_TO=${REPORT_TO:-none}
WANDB_API_KEY=${WANDB_API_KEY:-${WANDB_KEY:-}}
if [[ "$USE_WANDB" == "True" || "$USE_WANDB" == "true" || "$REPORT_TO" == "wandb" ]]; then
  REPORT_TO="wandb"
  unset WANDB_DISABLED || true
  if [ -n "$WANDB_API_KEY" ]; then export WANDB_API_KEY; fi
else
  REPORT_TO="none"
  export WANDB_DISABLED=true
fi

# -------------------- 任务类型：分类任务用 classification loss，生成/阅读理解不用 --------------------
TRAIN_AS_CLS=True
case "$TASK_LOWER" in
  copa|record|recordd|record_task|record_dataset|drop|squad|squad_v1|squad1|winogrande|wikitext)
    TRAIN_AS_CLS=False ;;
  *)
    TRAIN_AS_CLS=True ;;
esac

EXTRA_ARGS=()
[ -n "$MODEL_PATH" ] && EXTRA_ARGS+=(--model_path "$MODEL_PATH")
[ "$LOAD_FLOAT16" = "True" ] && EXTRA_ARGS+=(--load_float16 True)
[ "$LOAD_BFLOAT16" = "True" ] && EXTRA_ARGS+=(--load_bfloat16 True)
[ "$LOAD_INT8" = "True" ] && EXTRA_ARGS+=(--load_int8 True)
[ "$LOAD_INT4" = "True" ] && EXTRA_ARGS+=(--load_int4 True)
[ "$GRADIENT_CHECKPOINTING" = "True" ] && EXTRA_ARGS+=(--gradient_checkpointing True)
[ "$FORCE_DISABLE_BNB" = "True" ] && EXTRA_ARGS+=(--force_disable_bnb True)
[ "$NO_OUTLIER" = "True" ] && EXTRA_ARGS+=(--no_outlier True)
EXTRA_ARGS+=(--prefer_local_model "$PREFER_LOCAL_MODEL" --max_length "$MAX_LENGTH" --report_to "$REPORT_TO")

RUN_NAME="$TASK-${MODEL_NAME}-${PRECISION_PROFILE:-manual_W${WBIT}A${ABIT}}-altA${ALT_A_STEPS}B${ALT_B_STEPS}-r${LOQZO_RANK}-lr${LR}"
[ -n "$TAG" ] && RUN_NAME="$TAG-$RUN_NAME"
[ "$LOQZO_ADAPTIVE_RANK" = "True" ] && RUN_NAME="$RUN_NAME-adarank"
[ "$QZO_SCALE_SCOPE" != "weight" ] && RUN_NAME="$RUN_NAME-qzoscope${QZO_SCALE_SCOPE}"
RUN_DIR="$LOG_HOME/$RUN_NAME"
mkdir -p "$RUN_DIR"
if [ -n "$SAFE_LOG_SUFFIX" ]; then
  LOG_FILE="$RUN_DIR/${TIME_TAG}_${LOG_PREFIX}_${SAFE_LOG_SUFFIX}.log"
else
  LOG_FILE="$RUN_DIR/${TIME_TAG}_${LOG_PREFIX}.log"
fi
exec > >(tee -a "$LOG_FILE") 2>&1
export LOQZO_SHELL_TEE=1

if [ "$LAUNCH_MODE" = "auto" ]; then
  if [ "$NUM_PROC" -le 1 ]; then
    LAUNCH_MODE="single"
  else
    # ZO/LoQZO 大模型更常用 model_parallel；FO/LoRA 才更适合 DDP。
    LAUNCH_MODE="model_parallel"
  fi
fi

echo "============================================================"
echo "RUN_NAME   : $RUN_NAME"
echo "RUN_DIR    : $RUN_DIR"
echo "LOG_FILE   : $LOG_FILE"
echo "LOG_SUFFIX : ${SAFE_LOG_SUFFIX:-<无>}"
echo "MODEL      : $MODEL"
echo "MODEL_PATH : ${MODEL_PATH:-<未指定>}"
echo "TASK       : $TASK"
echo "RUN_PY     : $RUN_PY"
echo "LAUNCH_MODE: $LAUNCH_MODE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<系统默认>}"
echo "PYTHONPATH : $PYTHONPATH"
echo "BS(real)=$REAL_BS | BS(per_device)=$BS | GRAD_ACC=$GRAD_ACC"
echo "LR=$LR | EPS=$EPS | QZO_EPS=${QZO_EPS} | STEPS=$STEPS | EPOCHS=$EPOCHS"
echo "PRECISION_PROFILE=${PRECISION_PROFILE:-manual} | WBIT/ABIT/PBIT=$WBIT/$ABIT/$PBIT | QMODE=$QMODE | WMODE=$WMODE | AMODE=$AMODE"
echo "LOAD dtype: fp16=$LOAD_FLOAT16 bf16=$LOAD_BFLOAT16 int8=$LOAD_INT8 int4=$LOAD_INT4 | AUTO_BF16_FOR_LARGE_MODEL=$AUTO_BF16_FOR_LARGE_MODEL"
echo "MAX_LENGTH=$MAX_LENGTH | FAST_TASK_LENGTH=$FAST_TASK_LENGTH"
echo "ALT: LoQZO=$ALT_A_STEPS | QZO-scale=$ALT_B_STEPS | ALT_START=$ALT_START"
echo "LoQZO: rank=$LOQZO_RANK adaptive=$LOQZO_ADAPTIVE_RANK basis=$LOQZO_BASIS_INIT fast_addmm=$LOQZO_FAST_ADDMM cache_params=$CACHE_TRAINABLE_PARAMS"
echo "LoQZO lazy sampling: update_basis=$LOQZO_UPDATE_BASIS sample_U=every_step v_update_freq=$LOQZO_V_UPDATE_FREQ legacy_u_update_freq=$LOQZO_U_UPDATE_FREQ"
echo "LoQZO skip/fuse: fullspace_1d=$LOQZO_FULLSPACE_FOR_1D skip_non_subspace_2d=$LOQZO_SKIP_NON_SUBSPACE_2D cache_coeff=$LOQZO_CACHE_LOWRANK_COEFF fuse_restore_update=$LOQZO_FUSE_RESTORE_UPDATE"
echo "QZO-scale: scope=$QZO_SCALE_SCOPE lr_mult=$QZO_SCALE_LR_MULT layerwise=$QZO_LAYERWISE_SCALE_PERTURB clip=$CLIP_ZO_GRAD threshold=$QZO_CLIP_THRESHOLD cache_params=$QZO_CACHE_SCALE_PARAMS cache_delta=$QZO_CACHE_SCALE_DELTAS"
echo "QZO-scale clamp: min=$QZO_SCALE_MIN max=$QZO_SCALE_MAX max_mult=$QZO_SCALE_MAX_MULT"
echo "TRAIN/DEV/EVAL=$TRAIN/$DEV/$EVAL | NO_EVAL=$NO_EVAL | EVAL_DURING_TRAINING=$EVAL_DURING_TRAINING | SKIP_POST_UPDATE_FORWARD=$SKIP_POST_UPDATE_FORWARD"
echo "Speed extra: prepare_inputs_once=$PREPARE_INPUTS_ONCE_FOR_ZO fast_eval_mode=$FAST_ZO_EVAL_MODE disable_train_use_cache=$DISABLE_TRAIN_USE_CACHE qft_linear_only=$QFT_TRAIN_QUANTIZED_LINEAR_ONLY qft_train_bias=$QFT_TRAIN_BIAS"
echo "Speed v3: fast_option_loss=$FAST_OPTION_LOSS fast_eval_batch_options=$FAST_EVAL_BATCH_OPTIONS skip_flos=$SKIP_FLOS_METER_FOR_ZO disable_nan_inf_filter=$DISABLE_NAN_INF_FILTER_FOR_ZO fast_int_quant=$FAST_INT_QUANT no_outlier=$NO_OUTLIER eval_num_demos=$EVAL_NUM_DEMOS"
echo "============================================================"

COMMON_ARGS=(
  --model_name "$MODEL" "${GPU_ARGS[@]}" --task_name "$TASK" --output_dir "$RUN_DIR" --run_name "$RUN_NAME"
  --trainer "$TRAINER" --trainer_module "$TRAINER_MODULE" --tuning_type "$TYPE"
  --train_set_seed "$SEED" --num_train "$TRAIN" --num_dev "$DEV" --num_eval "$EVAL"
  --logging_steps "$LOGGING_STEPS" --gradient_accumulation_steps "$GRAD_ACC" --optim "$OPTIM"
  --learning_rate "$LR" --zo_eps "$EPS" --per_device_train_batch_size "$BS"
  --skip_post_update_forward "$SKIP_POST_UPDATE_FORWARD" --cache_trainable_params "$CACHE_TRAINABLE_PARAMS"
  --loqzo_fast_addmm "$LOQZO_FAST_ADDMM" --qzo_cache_scale_params "$QZO_CACHE_SCALE_PARAMS" --qzo_cache_scale_deltas "$QZO_CACHE_SCALE_DELTAS"
  --prepare_inputs_once_for_zo "$PREPARE_INPUTS_ONCE_FOR_ZO" --fast_zo_eval_mode "$FAST_ZO_EVAL_MODE" --disable_train_use_cache "$DISABLE_TRAIN_USE_CACHE"
  --qft_train_quantized_linear_only "$QFT_TRAIN_QUANTIZED_LINEAR_ONLY" --qft_train_bias "$QFT_TRAIN_BIAS"
  --fast_option_loss "$FAST_OPTION_LOSS" --fast_eval_batch_options "$FAST_EVAL_BATCH_OPTIONS"
  --skip_flos_meter_for_zo "$SKIP_FLOS_METER_FOR_ZO" --disable_nan_inf_filter_for_zo "$DISABLE_NAN_INF_FILTER_FOR_ZO" --fast_int_quant "$FAST_INT_QUANT"
  --save_strategy steps --save_steps "$SAVE_STEPS" --no_eval "$NO_EVAL"
  --quantized_perturb_ours "$TWO" --train_as_classification "$TRAIN_AS_CLS"
  --perturb_bits "$PBIT" --mask_ratio "$MASK_RATIO" --num_pertub "$NUM_PERTUB"
  --overwrite_output_dir "$OVERWRITE_OUTPUT_DIR" --use_eval_demos_after_training "$USE_EVAL_DEMOS_AFTER_TRAINING" --eval_num_demos "$EVAL_NUM_DEMOS" --eval_demo_seed "$EVAL_DEMO_SEED"
  --precision_profile "$PRECISION_PROFILE" --mode "$QMODE" --wmode "$WMODE" --amode "$AMODE" --wbit "$WBIT" --abit "$ABIT" --qft_freeze_alpha True --qft_alpha_only False
  --loqzo_enable "$LOQZO_ENABLE" --loqzo_rank "$LOQZO_RANK" --loqzo_adaptive_rank "$LOQZO_ADAPTIVE_RANK"
  --loqzo_rank_min "$LOQZO_RANK_MIN" --loqzo_rank_max "$LOQZO_RANK_MAX" --loqzo_rank_budget "$LOQZO_RANK_BUDGET"
  --loqzo_rank_update_freq "$LOQZO_RANK_UPDATE_FREQ" --loqzo_rank_ema "$LOQZO_RANK_EMA" --loqzo_basis_init "$LOQZO_BASIS_INIT"
  --loqzo_update_basis "$LOQZO_UPDATE_BASIS" --loqzo_v_update_freq "$LOQZO_V_UPDATE_FREQ" --loqzo_u_update_freq "$LOQZO_U_UPDATE_FREQ"
  --loqzo_update_v_every_step "$LOQZO_UPDATE_V_EVERY_STEP" --loqzo_u_refresh_mode "$LOQZO_U_REFRESH_MODE"
  --loqzo_include_embeddings "$LOQZO_INCLUDE_EMBEDDINGS" --loqzo_fullspace_for_1d "$LOQZO_FULLSPACE_FOR_1D"
  --loqzo_skip_non_subspace_2d "$LOQZO_SKIP_NON_SUBSPACE_2D" --loqzo_cache_lowrank_coeff "$LOQZO_CACHE_LOWRANK_COEFF" --loqzo_fuse_restore_update "$LOQZO_FUSE_RESTORE_UPDATE"
  --loqzo_quantize_coeff "$LOQZO_QUANTIZE_COEFF" --loqzo_coeff_bits "$LOQZO_COEFF_BITS"
  --alt_a_steps "$ALT_A_STEPS" --alt_b_steps "$ALT_B_STEPS" --alt_start "$ALT_START"
  --qzo_eps "$QZO_EPS" --qzo_scale_lr_mult "$QZO_SCALE_LR_MULT" --qzo_scale_min "$QZO_SCALE_MIN"
  --qzo_scale_max "$QZO_SCALE_MAX" --qzo_scale_max_mult "$QZO_SCALE_MAX_MULT"
  --qzo_scale_scope "$QZO_SCALE_SCOPE" --qzo_layerwise_scale_perturb "$QZO_LAYERWISE_SCALE_PERTURB"
  --clip_zo_grad "$CLIP_ZO_GRAD" --qzo_clip_threshold "$QZO_CLIP_THRESHOLD" --qzo_require_qft True
)
[ -n "$LOQZO_TARGET_MODULES" ] && COMMON_ARGS+=(--loqzo_target_modules "$LOQZO_TARGET_MODULES")
if [ "$EVAL_DURING_TRAINING" = "True" ]; then
  COMMON_ARGS+=(--evaluation_strategy steps --eval_steps "$EVAL_STEPS")
else
  COMMON_ARGS+=(--evaluation_strategy no)
fi
if [ "$EPOCHS" != "0" ] && [ "$EPOCHS" != "0.0" ]; then
  COMMON_ARGS+=(--num_train_epochs "$EPOCHS" --max_steps -1)
else
  COMMON_ARGS+=(--max_steps "$STEPS")
fi
[ -n "$RESUME_FROM_CHECKPOINT" ] && COMMON_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
COMMON_ARGS+=("${EXTRA_ARGS[@]}")

run_single() {
  WANDB_PROJECT="$WANDB_PROJECT" python "$RUN_PY" "${COMMON_ARGS[@]}" "$@"
}
run_torchrun() {
  WANDB_PROJECT="$WANDB_PROJECT" torchrun --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" \
    --node_rank "$NODE_RANK" --master_port "$MASTER_PORT" $TORCHRUN_EXTRA_ARGS \
    "$RUN_PY" --distributed True "${COMMON_ARGS[@]}" "$@"
}
run_zero3() {
  WANDB_PROJECT="$WANDB_PROJECT" torchrun --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" \
    --node_rank "$NODE_RANK" --master_port "$MASTER_PORT" $TORCHRUN_EXTRA_ARGS \
    "$RUN_PY" --distributed True --deepspeed "$DEEPSPEED_CONFIG" "${COMMON_ARGS[@]}" "$@"
}

case "$LAUNCH_MODE" in
  single|model_parallel) run_single "$@" ;;
  ddp) run_torchrun "$@" ;;
  zero3) run_zero3 "$@" ;;
  *) echo "不支持的 LAUNCH_MODE: $LAUNCH_MODE"; exit 1 ;;
esac

# ========================= 运行命令示例 =========================
# 1) 单卡 OPT-1.3B / SST2 / W4A8 / 1:1 交替，按步数训练：
# GPU_ID=0 TASK=SST2 MODEL=facebook/opt-1.3b STEPS=5000 WBIT=4 ABIT=8 ALT_A_STEPS=1 ALT_B_STEPS=1 LOQZO_RANK=8 bash Code/script/alternating_loqzo_qzo.sh --logname sst2_r8
#
# 2) 多卡模型并行，GPU_ID 填多个即可；auto 会切到 model_parallel：
# GPU_ID=0,1 TASK=BoolQ MODEL=facebook/opt-6.7b STEPS=5000 BS=8 WBIT=4 ABIT=8 bash Code/script/alternating_loqzo_qzo.sh --logname boolq_6.7b_2gpu
#
# 3) 按 epoch 训练；EPOCHS>0 时自动忽略 STEPS：
# GPU_ID=0 TASK=RTE MODEL=facebook/opt-1.3b EPOCHS=3 WBIT=4 ABIT=8 bash Code/script/alternating_loqzo_qzo.sh --logname rte_epoch3
#
# 4) 更偏 LoQZO 的 3:1 交替：
# GPU_ID=0 TASK=MultiRC MODEL=facebook/opt-1.3b ALT_A_STEPS=3 ALT_B_STEPS=1 LOQZO_RANK=8 BS=8 LR=1e-4 STEPS=5000 bash Code/script/alternating_loqzo_qzo.sh --logname multirc_a3b1
#
# 5) 先 LoQZO warmup 500 步，再 1:1 交替：
# GPU_ID=0 TASK=SST2 MODEL=facebook/opt-1.3b ALT_START=500 ALT_A_STEPS=1 ALT_B_STEPS=1 LOQZO_RANK=8 STEPS=10000 bash Code/script/alternating_loqzo_qzo.sh --logname warmup500
#
# 6) CB / Copa / WSC 这类小数据集无需手动设 DEV；脚本默认 DEV=0，避免训练集被切空：
# GPU_ID=0 TASK=CB MODEL=facebook/opt-1.3b STEPS=5000 WBIT=4 ABIT=8 bash Code/script/alternating_loqzo_qzo.sh --logname cb_default
#
# 7) ReCoRD / SQuAD / DROP：脚本会自动使用生成式/阅读理解训练设置；本地 List feature 兼容由 tasks.py 处理：
# GPU_ID=0 TASK=SQuAD MODEL=facebook/opt-1.3b STEPS=5000 WBIT=4 ABIT=8 bash Code/script/alternating_loqzo_qzo.sh --logname squad_default
# GPU_ID=0 TASK=ReCoRD MODEL=facebook/opt-1.3b STEPS=5000 WBIT=4 ABIT=8 bash Code/script/alternating_loqzo_qzo.sh --logname record_default
# GPU_ID=0 TASK=DROP MODEL=facebook/opt-1.3b STEPS=5000 WBIT=4 ABIT=8 bash Code/script/alternating_loqzo_qzo.sh --logname drop_default

# 7) 对齐 QuZO 表 1 的 Data Precision 跑我们自己的算法：
# FP / W32A32：全精度前向，无 scale，默认只跑 LoQZO。
# GPU_ID=0 TASK=SST2 MODEL=Llama2-7B PRECISION_PROFILE=FP_W32A32 STEPS=10000 SAVE_STEPS=50000 bash Code/script/alternating_loqzo_qzo.sh --logname sst2_fp_w32a32
#
# FP / W8A8：权重 float8，激活 float8。
# GPU_ID=0 TASK=SST2 MODEL=Llama2-7B PRECISION_PROFILE=FP_W8A8 STEPS=10000 SAVE_STEPS=50000 bash Code/script/alternating_loqzo_qzo.sh --logname sst2_fp_w8a8
# 

# GPU_ID=5 TASK=SQuAD MODEL=Llama2-7B LOQZO_RANK=64 ALT_A_STEPS=1 ALT_B_STEPS=1 SEED=0 LR=1e-5 EPS=1e-3 PRECISION_PROFILE=FP_W8A8 STEPS=10000 SAVE_STEPS=50000 bash Code/script/alternating_loqzo_qzo.sh --logname squad_fp_w8a8
# GPU_ID=0 TASK=MultiRC MODEL=Llama2-7B LOQZO_RANK=128 ALT_A_STEPS=4 ALT_B_STEPS=1 SEED=0 LR=1e-5 EPS=1e-3 PRECISION_PROFILE=FP_W8A8 STEPS=10000 SAVE_STEPS=50000 bash Code/script/alternating_loqzo_qzo.sh --logname MultiRC_128_lqazo
# GPU_ID=2 TASK=DROP MODEL=Llama2-7B LOQZO_RANK=128 ALT_A_STEPS=0 ALT_B_STEPS=1 SEED=0 LR=1e-5 EPS=1e-3 PRECISION_PROFILE=FP_W8A8 STEPS=10000 SAVE_STEPS=50000 bash Code/script/alternating_loqzo_qzo.sh --logname DROP_128_qzo
#
#主实验：INT W8A8
# GPU_ID=1 TASK=MultiRC MODEL=Llama2-7B LOQZO_RANK=128 ALT_A_STEPS=0 ALT_B_STEPS=1  SEED=0 LR=1e-5 EPS=1e-3 WBIT=8 ABIT=8 PRECISION_PROFILE=INT_W8A8   STEPS=10000 SAVE_STEPS=50000 bash  Code/script/alternating_loqzo_qzo.sh --logname QZO_MultiRC_int_w8a8
#
# INT / W8A8：权重 int8，激活 int8。
# GPU_ID=0 TASK=SST2 MODEL=Llama2-7B LOQZO_RANK=128 ALT_A_STEPS=4 ALT_B_STEPS=1 SEED=0 LR=1e-5 EPS=1e-3 PRECISION_PROFILE=INT_W8A8 STEPS=10000 SAVE_STEPS=50000 bash Code/script/alternating_loqzo_qzo.sh --logname sst2_lqazo_intw8a8
#
# INT/FP / W4A8：权重 int4，激活 float8。默认 profile 就是这个。
# GPU_ID=0 TASK=SST2 MODEL=Llama2-7B PRECISION_PROFILE=INTFP_W4A8 STEPS=10000 SAVE_STEPS=50000 bash Code/script/alternating_loqzo_qzo.sh --logname sst2_intfp_w4a8
#
#
#   INT/FP W4A4参数设定
# RANK=128 LQAZO(A=1 B=4) QZO(A=0 B=1) LR=1e-5 EPS=1e-3 
# 
# GPU_ID=4 TASK=SST2 MODEL=Llama2-7B LOQZO_RANK=128 ALT_A_STEPS=0 ALT_B_STEPS=1  SEED=0 LR=1e-5 EPS=1e-3  PRECISION_PROFILE= WBIT=4 ABIT=4 QMODE=int WMODE=int AMODE=float STEPS=10000 SAVE_STEPS=50000 bash  Code/script/alternating_loqzo_qzo.sh --logname QZO_SST2_w4a4
#