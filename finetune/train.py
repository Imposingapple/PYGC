#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.nn.modules import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer

# debug专用
### 开始
import sys
sys.path.append("/home/haoping/Projects/chinese_spell_checking/PYGC")
### 结束
from models.modeling import PYGC
from datas.bert_csc_dataset import TestCSCDataset, Dynaimic_CSCDataset, TestCSCDIMEDataset
from datas.collate_functions import collate_to_max_length_with_id, collate_to_max_length_for_train_dynamic_pron_loss
from utils.random_seed import set_random_seed


set_random_seed(2333)

def decode_sentence_and_get_pinyinids(ids):
    dataset = TestCSCDataset(
        data_path='../scope/data/test.sighan15.pkl',
        vocab_path='./datas/',
    )
    sent = ''.join(dataset.tokenizer.decode(ids).split(' '))
    src_pinyin, _ = dataset.hanzi2pinyin.convert_sentence_to_pinyin_ids(sent)
    pinyin_input_ids = dataset.pinyin_tokenier.encode(src_pinyin)
    pinyin_input_ids = torch.LongTensor(pinyin_input_ids).unsqueeze(0)
    return sent, pinyin_input_ids


class CSCTask(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        """Initialize a models, tokenizer and config."""
        super().__init__()
        self.args = args
        if isinstance(args, argparse.Namespace):
            self.save_hyperparameters(args)
        
        # 模型部分
        self.vocab_file = os.path.join(self.args.vocab_path, 'vocab.txt')
        self.tokenizer = BertTokenizer(self.vocab_file, do_lower_case=True)
        self.pinyin_vocab_file = os.path.join(self.args.vocab_path, 'vocab_pinyin.txt')
        self.pinyin_tokenizer = BertTokenizer(self.pinyin_vocab_file, do_lower_case=False)

        self.model = PYGC( self.tokenizer.vocab_size, 
                                self.args.pretrained_hanzi_lm,
                                self.pinyin_tokenizer.vocab_size, 
                                self.args.pretrained_pinyin_lm_file,
                                self.args.multitask
                                )

        if args.ckpt_path is not None: # resume from a checkpoint
            print("loading from ", args.ckpt_path)
            ckpt = torch.load(args.ckpt_path, )["state_dict"]
            # ckpt = torch.load(args.ckpt_path, map_location=torch.device('cpu'))["state_dict"]  # 在cpu上加载模型
            new_ckpt = {}
            for key in ckpt.keys():
                new_ckpt[key[6:]] = ckpt[key]
            self.model.load_state_dict(new_ckpt,strict=False)
            # print(self.model.device, torch.cuda.is_available())
            # quit()

        # criterions
        self.loss_fct = CrossEntropyLoss() # token/pinyin classification的损失函数
        gpus_string = (
            str(self.args.gpus) if not str(self.args.gpus).endswith(",") else str(self.args.gpus)[:-1]
        )
        self.num_gpus = len(gpus_string.split(","))

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.98),  # according to RoBERTa paper
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
        )
        t_total = (
            len(self.train_dataloader())
            // self.args.accumulate_grad_batches
            * self.args.max_epochs
        )
        warmup_steps = int(self.args.warmup_proporation * t_total)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def forward(self, input_ids, pinyin_input_ids):
        attention_mask = (input_ids != 0).long()
        pinyin_attention_mask = (pinyin_input_ids != 0).long()
        return self.model(input_ids, attention_mask, pinyin_input_ids, pinyin_attention_mask)

    def compute_loss(self, batch):
        input_ids, label_ids, det_labels, py_input_ids, \
            py_label_ids, py_det_labels = batch
        batch_size, length = input_ids.shape
        # print("batch_size, length: ", batch_size, length)
        cor_out, det_out, py_cor_out, py_det_out = self.forward(input_ids, py_input_ids)
        
        # 汉字loss
        loss_mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        active_loss = loss_mask.view(-1) == 1
        active_cor_labels = torch.where(
            active_loss, label_ids.view(-1), torch.tensor(self.loss_fct.ignore_index)
            .type_as(label_ids)
        )
        hanzi_cor_loss = self.loss_fct(cor_out.view(-1, self.tokenizer.vocab_size), active_cor_labels)
        hanzi_det_loss = 0
        if self.args.multitask:
            active_det_labels = torch.where(
                active_loss, det_labels.view(-1), torch.tensor(self.loss_fct.ignore_index)
                .type_as(det_labels)
            )
            hanzi_det_loss = self.loss_fct(det_out.view(-1, 2), active_det_labels)
        
        # 拼音loss
        py_loss_mask = (py_input_ids != 0) * (py_input_ids != 2) * (py_input_ids != 3).long()
        py_active_loss = py_loss_mask.view(-1) == 1
        py_active_cor_labels = torch.where(
            py_active_loss, py_label_ids.view(-1), torch.tensor(self.loss_fct.ignore_index)
            .type_as(py_label_ids)
        )
        pinyin_cor_loss = self.loss_fct(py_cor_out.view(-1, self.pinyin_tokenizer.vocab_size), 
                                        py_active_cor_labels)
        pinyin_det_loss = 0
        if self.args.multitask:
            py_active_det_labels = torch.where(
                py_active_loss, py_det_labels.view(-1), torch.tensor(self.loss_fct.ignore_index)
                .type_as(py_det_labels)
            )
            pinyin_det_loss = self.loss_fct(py_det_out.view(-1, 2), py_active_det_labels)

        return hanzi_cor_loss, hanzi_det_loss, pinyin_cor_loss, pinyin_det_loss

    def training_step(self, batch, batch_idx):
        """"""
        hanzi_cor_loss, hanzi_det_loss, pinyin_cor_loss, pinyin_det_loss = self.compute_loss(batch)
        c = self.args.c
        loss = (1 - c) * (hanzi_cor_loss + hanzi_det_loss) + c * (pinyin_cor_loss + pinyin_det_loss)
        tf_board_logs = {
            "total_loss": loss,
            "hanzi_cor_loss": hanzi_cor_loss,
            "pinyin_cor_loss": pinyin_cor_loss,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        # torch.cuda.empty_cache()
        self.log_dict(tf_board_logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        """"""
        input_ids, pinyin_input_ids, label_ids, pinyin_label_ids, ids, srcs, tokens_size = batch
        batch_size, length = input_ids.shape

        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        attention_mask = (input_ids != 0).long()
        pinyin_attention_mask = (pinyin_input_ids != 0).long()
        cor_out, det_out, _, _ = self.model(input_ids, 
                                      attention_mask, 
                                      pinyin_input_ids, 
                                      pinyin_attention_mask)
        predict_scores = F.softmax(cor_out, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask
        return {
            "tgt_idx": label_ids.cpu(),
            "pred_idx": predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }

    def validation_epoch_end(self, outputs):
        from metrics.metric import Metric

        # print(len(outputs))
        metric = Metric(vocab_path=self.args.vocab_path)
        pred_txt_path = os.path.join(self.args.save_path, "preds.txt")
        pred_lbl_path = os.path.join(self.args.save_path, "labels.txt")
        if len(outputs) == 2: # sanity check默认2个batch
            self.log("df", 0)
            self.log("cf", 0)
            return {"df": 0, "cf": 0}
        results = metric.metric(
            batches=outputs,
            pred_txt_path=pred_txt_path,
            pred_lbl_path=pred_lbl_path,
            label_path=self.args.label_file,
        )
        self.log("df", results["sent-detect-f1"])
        self.log("cf", results["sent-correct-f1"])
        return {"df": results["sent-detect-f1"], "cf": results["sent-correct-f1"]}

    def train_dataloader(self) -> DataLoader:
        name = "sighan/train_all"
        # name = "sighan/train_all_isolation"
        # name = "cscd-ime/cscd_trainset.json"

        # dataset的fields：input_ids, label_ids, det_labels, pinyin_input_ids, pinyin_label_ids, pinyin_det_labels
        dataset = Dynaimic_CSCDataset(
            data_path=os.path.join(self.args.data_dir, name),
            vocab_path=self.args.vocab_path,
            max_length=self.args.max_length,
        )
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = dataset.tokenizer

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_for_train_dynamic_pron_loss, fill_values=[0, 0, 0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def val_dataloader(self):
        dataset = TestCSCDataset(
            data_path='./data/sighan/test.sighan15.pkl',
            vocab_path=self.args.vocab_path,
            max_length=self.args.max_length,
        )
        # dataset = TestCSCDIMEDataset(
        #     data_path='./data/cscd-ime/dev.tsv',
        #     vocab_path=self.args.vocab_path,
        #     max_length=self.args.max_length,
        # )
        print('dev dataset', len(dataset))

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def test15_dataloader(self):
        dataset = TestCSCDataset(
            data_path='./data/sighan/test.sighan15.pkl',
            vocab_path=self.args.vocab_path,
            max_length=self.args.max_length,
        )

        self.tokenizer = dataset.tokenizer
        from datas.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader
    
    def test_cscd_dataloader(self):
        dataset = TestCSCDIMEDataset(
            data_path='./data/cscd-ime/test.tsv',
            vocab_path=self.args.vocab_path,
            max_length=self.args.max_length,
        )

        self.tokenizer = dataset.tokenizer
        from datas.collate_functions import collate_to_max_length_with_id

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=partial(collate_to_max_length_with_id, fill_values=[0, 0, 0, 0]),
            drop_last=False,
        )
        return dataloader

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids, pinyin_input_ids, label_ids, pinyin_label_ids, ids, srcs, tokens_size = batch
        batch_size, length = input_ids.shape

        mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
        attention_mask = (input_ids != 0).long()
        pinyin_attention_mask = (pinyin_input_ids != 0).long()
        cor_out, det_out, _, _ = self.model(input_ids, 
                                      attention_mask, 
                                      pinyin_input_ids, 
                                      pinyin_attention_mask)
        predict_scores = F.softmax(cor_out, dim=-1)
        predict_labels = torch.argmax(predict_scores, dim=-1) * mask

        if '13' in self.args.label_file:
            predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
        
        pre_predict_labels = predict_labels
        for _ in range(1):
            record_index = []
            for i,(a,b) in enumerate(zip(list(input_ids[0,1:-1]),list(predict_labels[0,1:-1]))):
                if a!=b:
                    record_index.append(i)
            
            input_ids[0,1:-1] = predict_labels[0,1:-1]
            sent, new_pinyin_ids = decode_sentence_and_get_pinyinids(input_ids[0,1:-1].cpu().numpy().tolist())
            pinyin_ids = new_pinyin_ids.to(input_ids.device)
            # print(input_ids.device, pinyin_ids.device)

            cor_out, det_out, _, _ = self.model(input_ids, 
                                      attention_mask, 
                                      pinyin_input_ids, 
                                      pinyin_attention_mask)
            predict_scores = F.softmax(cor_out, dim=-1)
            predict_labels = torch.argmax(predict_scores, dim=-1) * mask

            for i,(a,b) in enumerate(zip(list(input_ids[0,1:-1]),list(predict_labels[0,1:-1]))):
                if a!=b and any([abs(i-x)<=1 for x in record_index]):
                    print(ids,srcs)
                    print(i+1,)
                else:
                    predict_labels[0,i+1] = input_ids[0,i+1]
            if predict_labels[0,i+1] == input_ids[0,i+1]:
                break
            if '13' in self.args.label_file:
                predict_labels[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))] = \
                    input_ids[(predict_labels == self.tokenizer.token_to_id('地')) | (predict_labels == self.tokenizer.token_to_id('得'))]
        # if not pre_predict_labels.equal(predict_labels):
        #     print([self.tokenizer.id_to_token(id) for id in pre_predict_labels[0][1:-1]])
        #     print([self.tokenizer.id_to_token(id) for id in predict_labels[0][1:-1]])
        return {
            "tgt_idx": label_ids.cpu(),
            "post_pred_idx": predict_labels.cpu(),
            "pred_idx": pre_predict_labels.cpu(),
            "id": ids,
            "src": srcs,
            "tokens_size": tokens_size,
        }


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--data_dir", required=True, type=str, help="train data path")
    parser.add_argument(
        "--label_file",
        default="./data/sighan/test.sighan15.lbl.tsv",
        type=str,
        help="label file",
    )
    parser.add_argument(
        "--pretrained_hanzi_lm", 
        default='hfl/chinese-roberta-wwm-ext', 
        type=str
    )
    parser.add_argument(
        "--pretrained_pinyin_lm_file", 
        default='./checkpoints/pinyin_bert-finetuned-wikizh_electra/checkpoint-233130', 
        type=str
    )
    parser.add_argument("--save_path", required=True, type=str)
    parser.add_argument("--vocab_path", default="./datas/", help="path of vocab.txt & vocab_pinyin.txt")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument("--multitask", action="store_true", default=False, help="det and cor")
    parser.add_argument("--c", type=float, default=0.5, help="weight of pinyin loss")
    parser.add_argument(
        "--workers", type=int, default=8, help="num workers for dataloader"
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument(
        "--use_memory",
        action="store_true",
        help="load datasets to memory to accelerate.",
    )
    parser.add_argument(
        "--max_length", default=512, type=int, help="max length of datasets"
    )
    parser.add_argument("--checkpoint_path", type=str, help="train checkpoint")
    parser.add_argument(
        "--save_topk", default=5, type=int, help="save topk checkpoint"
    )
    parser.add_argument("--mode", default="train", type=str, help="train or evaluate")
    parser.add_argument(
        "--warmup_proporation", default=0.01, type=float, help="warmup proporation"
    )
    parser.add_argument("--gamma", default=1, type=float, help="phonetic loss weight")
    parser.add_argument(
        "--ckpt_path", default=None, type=str, help="resume_from_checkpoint"
    )
    return parser


def main():
    """main"""
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # create save path if doesn't exit
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model = CSCTask(args)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save_path, "checkpoint"),
        filename="{epoch}-{df:.4f}-{cf:.4f}",
        save_top_k=args.save_topk,
        monitor="cf",  # correction f1
        mode="max",
    )
    logger = TensorBoardLogger(save_dir=args.save_path, name="log")

    # save args
    if not os.path.exists(os.path.join(args.save_path, "checkpoint")):
        os.mkdir(os.path.join(args.save_path, "checkpoint"))
    with open(os.path.join(args.save_path, "args.json"), "w") as f:
        args_dict = args.__dict__
        del args_dict["tpu_cores"]
        json.dump(args_dict, f, indent=4)

    # num_sanity_val_steps默认为2，pt lightning先运行2个batch的val，方便调试
    trainer = Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback], logger=logger, 
        # num_sanity_val_steps=0  # 调试训练部分代码专用
    )

    # trainer.validate(model)
    trainer.fit(model)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
