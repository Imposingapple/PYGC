#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/home/haoping/Projects/chinese_spell_checking/PYGC
VOCAB_PATH=/home/haoping/Projects/chinese_spell_checking/PYGC/datas
DATA_DIR=/home/haoping/Projects/chinese_spell_checking/PYGC/data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=outputs/bs64epoch30/checkpoint/epoch=4-df=75.4683-cf=72.7922.ckpt
OUTPUT_DIR=outputs/bs64epoch30/predict_4
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=2 python -u finetune/predict.py \
  --ckpt_path $ckpt_path \
  --vocab_path $VOCAB_PATH \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file $DATA_DIR/sighan/test.sighan15.lbl.tsv \
  --gpus=0,