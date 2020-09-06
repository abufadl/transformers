#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"

python finetune.py --output_dir enro_finetune_baseline --tpu_cores 8 \
    --learning_rate=3e-5 \
    --gpus 0 \
    --do_train \
    --val_check_interval=0.25 \
    --adam_eps 1e-06 \
    --num_train_epochs 1 --src_lang en_XX --tgt_lang ro_RO \
    --data_dir $ENRO_DIR \
    --max_source_length $MAX_LEN --max_target_length $MAX_LEN --val_max_target_length $MAX_LEN --test_max_target_length $MAX_LEN \
    --train_batch_size=$BS --eval_batch_size=$BS \
    --task translation \
    --warmup_steps 500 \
    --freeze_embeds \
    --n_train 1000 --n_val 500 \
    --model_name_or_path=facebook/mbart-large-cc25 \
    "$@"
