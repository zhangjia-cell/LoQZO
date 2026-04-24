#!/usr/bin/env bash
set -euo pipefail

# MODEL: 模型名或本地路径。常用: facebook/opt-1.3b / facebook/opt-2.7b / facebook/opt-13b / facebook/opt-30b / meta-llama/Llama-2-7b-hf / meta-llama/Meta-Llama-3-8B / mistralai/Mistral-7B-v0.3
# MODEL_PATH: 显式本地模型路径（优先级最高）
# PREFER_LOCAL_MODEL: 是否优先从 Models/ 匹配本地模型；推荐 False，避免 1.3B 误匹配到 13B
# TASK: 任务名。常用: SST2 / RTE / BoolQ / CB / WIC / WSC / MultiRC / Copa / ReCoRD / SQuAD / DROP / WinoGrande / WikiText
# METHOD: 方法名。loqzo / loqzo_ft / quzo / mezo / fo
# MODE: 微调方式。ft / lora / prefix / loretta_rep / qft
# GPU_ID: 单卡编号；GPU_IDS: 多卡编号列表（如 0,1,2,3）
# LAUNCH_MODE: single / model_parallel / ddp / zero3 / auto
# BS: 每卡 batch size；GRAD_ACC: 梯度累积（FO 常用，ZO 通常建议 1）
# LR: 学习率；EPS: 零阶扰动步长
# STEPS: 总训练步数；EPOCHS>0 时改用 epoch 训练并忽略 STEPS
# LOG_HOME: 输出根目录；LOG_PREFIX: 日志文件前缀；TIME_TAG: 时间戳
# WBIT/ABIT/PBIT/QMODE: 量化相关参数
# LOAD_INT8/LOAD_INT4: 是否使用 8bit/4bit 底座加载（与 --wbit 不同）
# LOQZO_RANK: 固定 rank；LOQZO_ADAPTIVE_RANK: 是否自适应 rank；LOQZO_RANK_MIN/MAX/UPDATE_FREQ: 自适应 rank 范围与更新频率

export CUDA_HOME="${CUDA_HOME:-/home/zhangjia/cuda-12.1}"
export PATH="$CUDA_HOME/bin:$PATH"
SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
append_ld_path_if_exists() {
  local p="$1"
  if [ -n "$p" ] && [ -d "$p" ]; then
    case ":$LD_LIBRARY_PATH:" in *":$p:"*) ;; *) export LD_LIBRARY_PATH="$p${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;; esac
  fi
}
append_ld_path_if_exists "$CUDA_HOME/lib64"
append_ld_path_if_exists "$CUDA_HOME/targets/x86_64-linux/lib"
append_ld_path_if_exists "$SITE/nvidia/cuda_runtime/lib"
append_ld_path_if_exists "$SITE/nvidia/cusparse/lib"
append_ld_path_if_exists "$SITE/nvidia/nvjitlink/lib"
append_ld_path_if_exists "$SITE/nvidia/cublas/lib"
unset BNB_CUDA_VERSION

MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_PATH=${MODEL_PATH:-}
PREFER_LOCAL_MODEL=${PREFER_LOCAL_MODEL:-False}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
TASK=${TASK:-SST2}
RUN_PY=${RUN_PY:-Code/train/run_loqzo.py}
LOG_HOME=${LOG_HOME:-./outputs}
TAG=${TAG:-}
LOG_PREFIX=${LOG_PREFIX:-loqzo}
TIME_TAG=${TIME_TAG:-$(date +"%Y%m%d-%H%M%S")}

GPU_ID=${GPU_ID:-}
GPU_IDS=${GPU_IDS:-}
LAUNCH_MODE=${LAUNCH_MODE:-auto}
GPU_ARGS=()
NUM_PROC=1
if [ -n "$GPU_IDS" ]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  GPU_ARGS+=(--gpu_ids "$GPU_IDS")
  NUM_PROC=$(python - <<PY
s = "${GPU_IDS}"
print(len([x for x in s.split(',') if x.strip()]))
PY
)
elif [ -n "$GPU_ID" ]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  GPU_ARGS+=(--gpu_id "$GPU_ID")
fi
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=${MASTER_PORT:-29500}
NPROC_PER_NODE=${NPROC_PER_NODE:-$NUM_PROC}
TORCHRUN_EXTRA_ARGS=${TORCHRUN_EXTRA_ARGS:-}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-Code/script/ds_zero3_bf16.json}

BS=${BS:-16}
GRAD_ACC=${GRAD_ACC:-1}
REAL_BS=$((BS * GRAD_ACC))
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-10000}
EPOCHS=${EPOCHS:-0}
SAVE_STEPS=${SAVE_STEPS:-500}
LOGGING_STEPS=${LOGGING_STEPS:-20}
EVAL_STEPS=${EVAL_STEPS:-1000}
MAX_LENGTH=${MAX_LENGTH:-512}
NO_EVAL=${NO_EVAL:-False}
EVAL_DURING_TRAINING=${EVAL_DURING_TRAINING:-False}
OVERWRITE_OUTPUT_DIR=${OVERWRITE_OUTPUT_DIR:-False}
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
LOAD_BEST=${LOAD_BEST:-False}

METHOD=${METHOD:-loqzo}
TRAINER=${TRAINER:-$METHOD}
TRAINER_MODULE=${TRAINER_MODULE:-trainer_loqzo}
MODE=${MODE:-ft}
TYPE=${TYPE:-ft}
OPTIM=${OPTIM:-sgd}
NUM_PERTUB=${NUM_PERTUB:-1}
MASK_RATIO=${MASK_RATIO:-0}
TWO=${TWO:-True}

WBIT=${WBIT:-}
ABIT=${ABIT:-8}
PBIT=${PBIT:-4}
QMODE=${QMODE:-int}
LOAD_INT8=${LOAD_INT8:-False}
LOAD_INT4=${LOAD_INT4:-False}
LOAD_FLOAT16=${LOAD_FLOAT16:-False}
LOAD_BFLOAT16=${LOAD_BFLOAT16:-False}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
SMOOTH=${SMOOTH:-False}
QLLM=${QLLM:-False}
FO_QUANT_GRAD=${FO_QUANT_GRAD:-False}
FO_QUANT_BITS=${FO_QUANT_BITS:-8}
USE_GRAD_PRE_HOOK_QUANT=${USE_GRAD_PRE_HOOK_QUANT:-False}
GRAD_PRE_HOOK_BITS=${GRAD_PRE_HOOK_BITS:-8}
FORCE_DISABLE_BNB=${FORCE_DISABLE_BNB:-False}

LOQZO_ENABLE=${LOQZO_ENABLE:-True}
LOQZO_RANK=${LOQZO_RANK:-8}
LOQZO_ADAPTIVE_RANK=${LOQZO_ADAPTIVE_RANK:-False}
LOQZO_RANK_MIN=${LOQZO_RANK_MIN:-2}
LOQZO_RANK_MAX=${LOQZO_RANK_MAX:-64}
LOQZO_RANK_BUDGET=${LOQZO_RANK_BUDGET:-0}
LOQZO_RANK_UPDATE_FREQ=${LOQZO_RANK_UPDATE_FREQ:-200}
LOQZO_RANK_EMA=${LOQZO_RANK_EMA:-0.9}
LOQZO_BASIS_INIT=${LOQZO_BASIS_INIT:-random_orth}
LOQZO_TARGET_MODULES=${LOQZO_TARGET_MODULES:-}
LOQZO_INCLUDE_EMBEDDINGS=${LOQZO_INCLUDE_EMBEDDINGS:-False}
LOQZO_FULLSPACE_FOR_1D=${LOQZO_FULLSPACE_FOR_1D:-True}
LOQZO_QUANTIZE_COEFF=${LOQZO_QUANTIZE_COEFF:-True}
LOQZO_COEFF_BITS=${LOQZO_COEFF_BITS:-0}

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

EXTRA_ARGS=()
QUANT_ARGS=()
TRAIN_AS_CLS=True
case "$MODE" in
  prefix) TYPE="ft"; EXTRA_ARGS+=(--prefix_tuning True --num_prefix 5 --no_reparam True --prefix_init_by_real_act True) ;;
  lora)
    TYPE="lora"
    if [ "${WBIT:-}" = "8" ] && [ "$LOAD_INT4" != "True" ]; then LOAD_INT8=True; fi
    if [ "${WBIT:-}" = "4" ]; then LOAD_INT4=True; LOAD_INT8=False; fi
    ;;
  loretta_rep) TYPE="loretta_rep" ;;
  qft) TYPE="qft"; if [ -n "${WBIT:-}" ]; then QUANT_ARGS+=(--mode "$QMODE" --wbit "$WBIT" --abit "$ABIT"); fi ;;
esac
[ "$SMOOTH" = "True" ] && EXTRA_ARGS+=(--smooth True)
[ "$QLLM" = "True" ] && EXTRA_ARGS+=(--qllm True)
[ "$LOAD_BEST" = "True" ] && EXTRA_ARGS+=(--load_best_model_at_end True)
[ "$FORCE_DISABLE_BNB" = "True" ] && EXTRA_ARGS+=(--force_disable_bnb True)
[ "$LOAD_INT8" = "True" ] && EXTRA_ARGS+=(--load_int8 True)
[ "$LOAD_INT4" = "True" ] && EXTRA_ARGS+=(--load_int4 True)
[ "$LOAD_FLOAT16" = "True" ] && EXTRA_ARGS+=(--load_float16 True)
[ "$LOAD_BFLOAT16" = "True" ] && EXTRA_ARGS+=(--load_bfloat16 True)
[ "$GRADIENT_CHECKPOINTING" = "True" ] && EXTRA_ARGS+=(--gradient_checkpointing True)
[ -n "$MODEL_PATH" ] && EXTRA_ARGS+=(--model_path "$MODEL_PATH")
EXTRA_ARGS+=(--prefer_local_model "$PREFER_LOCAL_MODEL" --max_length "$MAX_LENGTH" --report_to "$REPORT_TO")

case "$TASK" in
  CB) DEV=100; TRAIN_AS_CLS=True ;;
  Copa) DEV=100; TRAIN_AS_CLS=False ;;
  ReCoRD|DROP|SQuAD|WinoGrande|WikiText) TRAIN_AS_CLS=False ;;
  *) TRAIN_AS_CLS=True ;;
esac

CLI_WBIT=""; CLI_ABIT=""; CLI_MODEL_PATH=""; CLI_PREFER_LOCAL=""; CLI_LOAD_INT8=""; CLI_LOAD_INT4=""
ARGS=("$@")
for ((i=0;i<${#ARGS[@]};i++)); do
  case "${ARGS[$i]}" in
    --wbit) CLI_WBIT="${ARGS[$((i+1))]:-}" ;;
    --abit) CLI_ABIT="${ARGS[$((i+1))]:-}" ;;
    --model_path) CLI_MODEL_PATH="${ARGS[$((i+1))]:-}" ;;
    --prefer_local_model) CLI_PREFER_LOCAL="${ARGS[$((i+1))]:-}" ;;
    --load_int8) CLI_LOAD_INT8="${ARGS[$((i+1))]:-}" ;;
    --load_int4) CLI_LOAD_INT4="${ARGS[$((i+1))]:-}" ;;
  esac
done
EFFECTIVE_WBIT="${CLI_WBIT:-$WBIT}"
EFFECTIVE_ABIT="${CLI_ABIT:-$ABIT}"
EFFECTIVE_MODEL_PATH="${CLI_MODEL_PATH:-$MODEL_PATH}"
EFFECTIVE_PREFER_LOCAL="${CLI_PREFER_LOCAL:-$PREFER_LOCAL_MODEL}"
EFFECTIVE_LOAD_INT8="${CLI_LOAD_INT8:-$LOAD_INT8}"
EFFECTIVE_LOAD_INT4="${CLI_LOAD_INT4:-$LOAD_INT4}"

RUN_NAME="$TASK-${MODEL_NAME}-m${METHOD}-mode${MODE}-bs${REAL_BS}-lr${LR}"
[ -n "$TAG" ] && RUN_NAME="$TAG-$RUN_NAME"
[ -n "${EFFECTIVE_WBIT:-}" ] && RUN_NAME="$RUN_NAME-W${EFFECTIVE_WBIT}"
[ -n "$PBIT" ] && RUN_NAME="$RUN_NAME-P${PBIT}"
RUN_NAME="$RUN_NAME-r${LOQZO_RANK}"
[ "$MODE" = "lora" ] && RUN_NAME="$RUN_NAME-lora"
[ "$MODE" = "qft" ] && RUN_NAME="$RUN_NAME-qft-${QMODE}"
[ "$SMOOTH" = "True" ] && RUN_NAME="$RUN_NAME-smooth"
[ "$QLLM" = "True" ] && RUN_NAME="$RUN_NAME-qllm"
RUN_DIR="$LOG_HOME/$RUN_NAME"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/${TIME_TAG}_${LOG_PREFIX}.log"
exec > >(tee -a "$LOG_FILE") 2>&1
export LOQZO_SHELL_TEE=1

if [ "$LAUNCH_MODE" = "auto" ]; then
  if [ "$NUM_PROC" -le 1 ]; then LAUNCH_MODE="single"; else
    case "$METHOD" in fo) LAUNCH_MODE="ddp" ;; *) LAUNCH_MODE="model_parallel" ;; esac
  fi
fi

TRAINER_ECHO="$TRAINER"
case "${TRAINER,,}" in loqzo) TRAINER_ECHO="zo_lowbit" ;; loqzo_ft) TRAINER_ECHO="zo_lowbit_ft" ;; quzo) TRAINER_ECHO="zo_lowbit" ;; quzo_ft) TRAINER_ECHO="zo_lowbit_ft" ;; mezo) TRAINER_ECHO="zo" ;; fo) TRAINER_ECHO="regular" ;; esac

echo "============================================================"
echo "RUN_NAME   : $RUN_NAME"
echo "RUN_DIR    : $RUN_DIR"
echo "LOG_FILE   : $LOG_FILE"
echo "MODEL      : $MODEL"
echo "MODEL_PATH : ${EFFECTIVE_MODEL_PATH:-<未指定>}"
echo "PREFER_LOCAL_MODEL: $EFFECTIVE_PREFER_LOCAL"
echo "TASK       : $TASK"
echo "METHOD/TRAINER: $METHOD / $TRAINER_ECHO"
echo "MODE/TYPE  : $MODE / $TYPE"
echo "TRAINER_MODULE: $TRAINER_MODULE"
echo "LAUNCH_MODE: $LAUNCH_MODE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<系统默认>}"
echo "BS(real)=$REAL_BS | BS(per_device)=$BS | GRAD_ACC=$GRAD_ACC"
echo "LR=$LR | EPS=$EPS | MAX_STEPS=$STEPS | EPOCHS=$EPOCHS | SAVE_STEPS=$SAVE_STEPS | LOGGING_STEPS=$LOGGING_STEPS"
echo "TRAIN/DEV/EVAL=$TRAIN/$DEV/$EVAL | NO_EVAL=$NO_EVAL | EVAL_DURING_TRAINING=$EVAL_DURING_TRAINING"
echo "WBIT/ABIT/PBIT=${EFFECTIVE_WBIT:-None}/${EFFECTIVE_ABIT}/${PBIT} | QMODE=$QMODE"
echo "LOAD_BFLOAT16/LOAD_FLOAT16/LOAD_INT8/LOAD_INT4=$LOAD_BFLOAT16/$LOAD_FLOAT16/$EFFECTIVE_LOAD_INT8/$EFFECTIVE_LOAD_INT4"
echo "LoQZO: enable=$LOQZO_ENABLE rank=$LOQZO_RANK adaptive=$LOQZO_ADAPTIVE_RANK rank_range=[$LOQZO_RANK_MIN,$LOQZO_RANK_MAX]"
echo "LOG_PREFIX=$LOG_PREFIX | TIME_TAG=$TIME_TAG"
echo "OVERWRITE_OUTPUT_DIR=$OVERWRITE_OUTPUT_DIR | RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-<未指定>}"
echo "WANDB: USE_WANDB=$USE_WANDB | REPORT_TO=$REPORT_TO | WANDB_DISABLED=${WANDB_DISABLED:-<unset>}"
echo "============================================================"

COMMON_ARGS=(
  --model_name "$MODEL" "${GPU_ARGS[@]}" --task_name "$TASK" --output_dir "$RUN_DIR" --run_name "$RUN_NAME"
  --trainer "$TRAINER" --trainer_module "$TRAINER_MODULE" --train_set_seed "$SEED" --num_train "$TRAIN" --num_dev "$DEV" --num_eval "$EVAL"
  --logging_steps "$LOGGING_STEPS" --gradient_accumulation_steps "$GRAD_ACC" --optim "$OPTIM" --learning_rate "$LR" --zo_eps "$EPS" --per_device_train_batch_size "$BS"
  --save_strategy steps --save_steps "$SAVE_STEPS" --no_eval "$NO_EVAL" --quantized_perturb_ours "$TWO" --train_as_classification "$TRAIN_AS_CLS" --perturb_bits "$PBIT"
  --tuning_type "$TYPE" --mask_ratio "$MASK_RATIO" --num_pertub "$NUM_PERTUB" --fo_quant_grad "$FO_QUANT_GRAD" --fo_quant_bits "$FO_QUANT_BITS" --use_grad_pre_hook_quant "$USE_GRAD_PRE_HOOK_QUANT" --grad_pre_hook_bits "$GRAD_PRE_HOOK_BITS"
  --overwrite_output_dir "$OVERWRITE_OUTPUT_DIR" --use_eval_demos_after_training True --eval_num_demos 32 --eval_demo_seed 0
  --loqzo_enable "$LOQZO_ENABLE" --loqzo_rank "$LOQZO_RANK" --loqzo_adaptive_rank "$LOQZO_ADAPTIVE_RANK" --loqzo_rank_min "$LOQZO_RANK_MIN" --loqzo_rank_max "$LOQZO_RANK_MAX" --loqzo_rank_budget "$LOQZO_RANK_BUDGET" --loqzo_rank_update_freq "$LOQZO_RANK_UPDATE_FREQ" --loqzo_rank_ema "$LOQZO_RANK_EMA" --loqzo_basis_init "$LOQZO_BASIS_INIT" --loqzo_include_embeddings "$LOQZO_INCLUDE_EMBEDDINGS" --loqzo_fullspace_for_1d "$LOQZO_FULLSPACE_FOR_1D" --loqzo_quantize_coeff "$LOQZO_QUANTIZE_COEFF" --loqzo_coeff_bits "$LOQZO_COEFF_BITS"
)
[ -n "$LOQZO_TARGET_MODULES" ] && COMMON_ARGS+=(--loqzo_target_modules "$LOQZO_TARGET_MODULES")
if [ "$EVAL_DURING_TRAINING" = "True" ]; then COMMON_ARGS+=(--evaluation_strategy steps --eval_steps "$EVAL_STEPS"); else COMMON_ARGS+=(--evaluation_strategy no); fi
if [ "$EPOCHS" != "0" ] && [ "$EPOCHS" != "0.0" ]; then COMMON_ARGS+=(--num_train_epochs "$EPOCHS" --max_steps -1); else COMMON_ARGS+=(--max_steps "$STEPS"); fi
[ -n "$RESUME_FROM_CHECKPOINT" ] && COMMON_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
[ "$LOAD_BEST" = "True" ] && [ "$EVAL_DURING_TRAINING" = "True" ] && COMMON_ARGS+=(--load_best_model_at_end True)
COMMON_ARGS+=("${EXTRA_ARGS[@]}" "${QUANT_ARGS[@]}")

run_single() { WANDB_PROJECT="$WANDB_PROJECT" python "$RUN_PY" "${COMMON_ARGS[@]}" "$@"; }
run_torchrun() { WANDB_PROJECT="$WANDB_PROJECT" torchrun --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" --node_rank "$NODE_RANK" --master_port "$MASTER_PORT" $TORCHRUN_EXTRA_ARGS "$RUN_PY" --distributed True "${COMMON_ARGS[@]}" "$@"; }
run_zero3() { WANDB_PROJECT="$WANDB_PROJECT" torchrun --nnodes "$NNODES" --nproc_per_node "$NPROC_PER_NODE" --node_rank "$NODE_RANK" --master_port "$MASTER_PORT" $TORCHRUN_EXTRA_ARGS "$RUN_PY" --distributed True --deepspeed "$DEEPSPEED_CONFIG" "${COMMON_ARGS[@]}" "$@"; }
case "$LAUNCH_MODE" in
  single|model_parallel) run_single "$@" ;;
  ddp) run_torchrun "$@" ;;
  zero3) run_zero3 "$@" ;;
  *) echo "不支持的 LAUNCH_MODE: $LAUNCH_MODE"; exit 1 ;;
esac

# ========================= 运行命令示例 =========================
# 可换模型: facebook/opt-1.3b | facebook/opt-2.7b | facebook/opt-13b | facebook/opt-30b | meta-llama/Llama-2-7b-hf | meta-llama/Llama-2-13b-hf | meta-llama/Meta-Llama-3-8B | mistralai/Mistral-7B-v0.3
# 可换任务: SST2 | RTE | BoolQ | CB | WIC | WSC | MultiRC | Copa | ReCoRD | SQuAD | DROP | WinoGrande | WikiText
# 可换方法: loqzo | loqzo_ft | quzo | mezo | fo
#
# 1) LoQZO 主实验（固定 rank）
# GPU_ID=0 METHOD=loqzo MODE=ft LOQZO_RANK=8 TASK=SST2 MODEL=facebook/opt-1.3b TAG=loqzo_sst2_r8 TRAIN=1000 DEV=500 EVAL=1000 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh --wbit 4 --abit 8
#
# 2) 对照组 QuZO
# GPU_ID=0 METHOD=quzo MODE=ft TASK=SST2 MODEL=facebook/opt-1.3b TAG=quzo_sst2 TRAIN=1000 DEV=500 EVAL=1000 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh --wbit 4 --abit 8
#
# 3) 对照组 MeZO
# GPU_ID=0 METHOD=mezo MODE=ft TASK=MultiRC MODEL=facebook/opt-1.3b TAG=mezo_multirc TRAIN=1000 DEV=500 EVAL=1000 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh
#
# 4) 对照组 FO
# GPU_ID=0 METHOD=fo MODE=ft TASK=SST2 MODEL=facebook/opt-1.3b TAG=fo_sst2 TRAIN=1000 DEV=500 EVAL=1000 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 LOAD_BFLOAT16=True GRADIENT_CHECKPOINTING=True PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh
#
# 5) 自适应 rank（LoQZO）
# GPU_ID=0 METHOD=loqzo MODE=ft LOQZO_RANK=8 LOQZO_ADAPTIVE_RANK=True LOQZO_RANK_MIN=2 LOQZO_RANK_MAX=32 LOQZO_RANK_UPDATE_FREQ=200 TASK=MultiRC MODEL=facebook/opt-1.3b TAG=loqzo_multirc_adapt TRAIN=1000 DEV=500 EVAL=1000 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh --wbit 4 --abit 8
#
# 6) LoRA（全精度底座）
# GPU_ID=0 METHOD=loqzo MODE=lora LOQZO_RANK=8 TASK=MultiRC MODEL=facebook/opt-1.3b TAG=loqzo_lora TRAIN=1000 DEV=500 EVAL=1000 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 FORCE_DISABLE_BNB=True PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh
#
# 7) LoRA + 8bit/4bit 底座（需 bitsandbytes 环境正常）
# GPU_ID=0 METHOD=loqzo MODE=lora LOQZO_RANK=8 WBIT=8 TASK=SQuAD MODEL=meta-llama/Meta-Llama-3-8B TAG=loqzo_lora_int8 TRAIN=1000 DEV=500 EVAL=1000 STEPS=3000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh
#
# 8) qft-int
# GPU_ID=0 METHOD=fo MODE=qft WBIT=4 ABIT=8 QMODE=int TASK=BoolQ MODEL=facebook/opt-1.3b TAG=qft_int TRAIN=1000 DEV=500 EVAL=1000 STEPS=1000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=256 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh
#
# 9) qft-float
# GPU_ID=0 METHOD=fo MODE=qft WBIT=4 ABIT=8 QMODE=float TASK=BoolQ MODEL=facebook/opt-1.3b TAG=qft_float TRAIN=1000 DEV=500 EVAL=1000 STEPS=1000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=256 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh
#
# 10) 多卡 model_parallel（适合 LoQZO/QuZO 大模型）
# GPU_IDS=0,1,2,3 LAUNCH_MODE=model_parallel METHOD=loqzo MODE=ft LOQZO_RANK=8 TASK=MultiRC MODEL=facebook/opt-13b TAG=loqzo_mp TRAIN=256 DEV=64 EVAL=128 BS=1 STEPS=500 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=256 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh --wbit 4 --abit 8
#
# 11) 多卡 DDP（适合 FO / LoRA）
# GPU_IDS=0,1,2,3 LAUNCH_MODE=ddp METHOD=fo MODE=ft TASK=SST2 MODEL=facebook/opt-1.3b TAG=fo_ddp TRAIN=1000 DEV=500 EVAL=1000 BS=2 GRAD_ACC=8 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 LOAD_BFLOAT16=True GRADIENT_CHECKPOINTING=True PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh
#
# 12) 断点续训
# GPU_ID=0 METHOD=loqzo MODE=ft LOQZO_RANK=8 TASK=SST2 MODEL=facebook/opt-1.3b TAG=loqzo_resume TRAIN=1000 DEV=500 EVAL=1000 STEPS=5000 SAVE_STEPS=500 LOGGING_STEPS=20 MAX_LENGTH=512 RESUME_FROM_CHECKPOINT=./outputs/<run_name>/checkpoint-1500 PREFER_LOCAL_MODEL=False LOG_HOME=./outputs bash Code/script/loqzo.sh --wbit 4 --abit 8
