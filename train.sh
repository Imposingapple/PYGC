#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/home/haoping/Projects/chinese_spell_checking/PYGC
VOCAB_PATH=/home/haoping/Projects/chinese_spell_checking/PYGC/datas
PRETRAINED_LM_FILE=/home/haoping/Projects/chinese_spell_checking/PYGC/checkpoints/pinyin_bert-finetuned-wikizh_electra/checkpoint-233130
DATA_DIR=$REPO_PATH/data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


lr=5e-5
bs=16
accumulate_grad_batches=4
epoch=30
# sighan
# OUTPUT_DIR=$REPO_PATH/outputs/sighan/bs$((bs*accumulate_grad_batches))epoch${epoch}
# sighan-isolation
OUTPUT_DIR=$REPO_PATH/outputs/sighan_isolation/bs$((bs*accumulate_grad_batches))epoch${epoch}  

mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=1 python -u $REPO_PATH/finetune/train.py \
--vocab_path $VOCAB_PATH \
--data_dir $DATA_DIR \
--pretrained_pinyin_lm_file $PRETRAINED_LM_FILE \
--save_path $OUTPUT_DIR \
--max_epoch $epoch \
--lr $lr \
--warmup_proporation 0.1 \
--batch_size=$bs \
--gpus=0, \
--accumulate_grad_batches=$accumulate_grad_batches  \
--reload_dataloaders_every_n_epochs 1 \
--precision 16 \
--save_topk 5 \
--ckpt_path ./checkpoints/pretrained_wiki.ckpt \

sleep 1

# nohup bash train.sh 2>&1 >train.log &