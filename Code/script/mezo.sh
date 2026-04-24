MODEL=${MODEL:-facebook/opt-1.3b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

BS=${BS:-16}
LR=${LR:-1e-5}
EPS=${EPS:-1e-3}
SEED=${SEED:-0}
TRAIN=${TRAIN:-1000}
DEV=${DEV:-500}
EVAL=${EVAL:-1000}
STEPS=${STEPS:-20000}
EVAL_STEPS=${EVAL_STEPS:-6000}
SAVE_STEPS=${SAVE_STEPS:-$EVAL_STEPS}
MASK_RATIO=${MASK_RATIO:-50}  # Default mask ratio of 50%
NUM_PERTUB=${NUM_PERTUB:-3}
TRAINER=${TRAINER:-zo}
OPTIM=${OPTIM:-sgd}
GRAD_ACC=${GRAD_ACC:-1}

REAL_BS=$((BS * GRAD_ACC))
MODE=${MODE:-ft}

EXTRA_ARGS=""
if [ "$MODE" == "prefix" ]; then
    EXTRA_ARGS="--prefix_tuning --num_prefix 5 --no_reparam --prefix_init_by_real_act"
elif [ "$MODE" == "lora" ]; then
    TYPE="lora"
    if [ "$WBIT" -eq 8 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --load_int8 True"
    elif [ "$WBIT" -eq 4 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --load_int4 True"
    fi
elif [ "$MODE" == "loretta_rep" ]; then
    TYPE="loretta_rep"
elif [ "$MODE" == "qft" ]; then
    TYPE="qft"
    if [ "$WBIT" -eq 4 ]; then
        QMODE=${QMODE:-int}
        EXTRA_ARGS="$EXTRA_ARGS --wbit $WBIT --mode $QMODE"
    fi
fi
if [[ ! -z "$LOAD_BEST" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --load_best_model_at_end"
fi
# if [[ ! -z "$CHECKPOINT" ]]; then
#     EXTRA_ARGS="$EXTRA_ARGS --resume_from_checkpoint $CHECKPOINT"
# fi

FP16=${FP16:-False} # When not quantizing, using bf16 in default
QUANT=${QUANT:-True}
QUANT_ARGS=""
# if [ "$QUANT" == "True" ]
# then
#     QMODE=${QMODE:-int}
#     WBIT=${WBIT:-8}
#     ABIT=${ABIT:-8}
#     QUANT_ARGS="$QUANT_ARGS --mode $QMODE --wbit $WBIT --abit $ABIT"
# fi



TASK_ARGS=""
case $TASK in
    # For Copa, ReCoRD, SQuAD, DROP, we set --train_as_classification False; for others, set this flag to True
    CB) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        ;;
    Copa) # It has <1000 training examples. Only use 100 for dev
        DEV=100
        TASK_ARGS="--train_as_classification False"
        ;;
    ReCoRD) 
        TASK_ARGS="--train_as_classification False"
        ;;
    DROP) 
        TASK_ARGS="--train_as_classification False"
        ;;
    SQuAD)
        TASK_ARGS="--train_as_classification False"
        ;;
    WinoGrande)
        TASK_ARGS="--train_as_classification False"
        ;;
    WikiText)
        TASK_ARGS="--train_as_classification False"
        ;;
esac

RUN_NAME="$TASK-${MODEL_NAME}-$REAL_BS-$LR-TWOPERTURB-$TWO"
if [ ! -z "$TAG" ]
then
    RUN_NAME="$TAG-$RUN_NAME"
fi
if [ ! -z "$WBIT" ]
then
    RUN_NAME="$RUN_NAME-W$WBIT-Perturb$PBIT-Bits"
fi




echo $RUN_NAME
echo "BS: $REAL_BS"
echo "LR: $LR"
echo "EPS: $EPS"
echo "SEED: $SEED"
echo "TRAIN/EVAL STEPS: $STEPS/$EVAL_STEPS"
echo "SAVE STEPS: $SAVE_STEPS"
echo "MODE: $MODE"
echo "Extra args: $EXTRA_ARGS $TASK_ARGS"
echo "WBIT: $WBIT"
echo "PBIT: $PBIT"
echo "QUANT: $QUANT"
echo "TWO: $TWO"




WANDB_PROJECT=${WANDB_PROJECT:-LLM_QAT_Perturb}

WANDB_PROJECT=$WANDB_PROJECT python run_mezo.py \
    --model_name $MODEL \
    --task_name $TASK \
    --output_dir $LOG_HOME/$RUN_NAME --run_name $RUN_NAME --train_set_seed $SEED --num_train $TRAIN --num_dev $DEV --num_eval $EVAL --logging_steps 10 \
    --max_steps $STEPS \
    --trainer $TRAINER --gradient_accumulation_steps $GRAD_ACC --optim $OPTIM \
    --learning_rate $LR --zo_eps $EPS --per_device_train_batch_size $BS --lr_scheduler_type "linear" \
    --evaluation_strategy steps --save_strategy steps\
    --eval_steps $EVAL_STEPS --save_steps $SAVE_STEPS --save_total_limit 1 --do_eval False \
    --train_as_classification \
    $QUANT_ARGS \
    $EXTRA_ARGS \
    $TASK_ARGS \
    "$@"
