#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/home/haoping/Projects/chinese_spell_checking/PYGC
VOCAB_PATH=/home/haoping/Projects/chinese_spell_checking/PYGC/datas
DATA_DIR=/home/haoping/Projects/chinese_spell_checking/PYGC/data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

# for sighan
# ckpt_path=outputs/finetuned/sighan/finetuned_checkpoint.ckpt
# OUTPUT_DIR=outputs/finetuned/sighan/predict
# for sighan_isolation
ckpt_path=outputs/finetuned/sighan_isolation/epoch=0-df=62.3701-cf=56.3410.ckpt
OUTPUT_DIR=outputs/finetuned/sighan_isolation/predict

mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=2 python -u finetune/predict.py \
  --ckpt_path $ckpt_path \
  --vocab_path $VOCAB_PATH \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file $DATA_DIR/sighan/test.sighan15.lbl.tsv \
  --gpus=0,
