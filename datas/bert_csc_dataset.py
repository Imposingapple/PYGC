#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
import pickle
from datas.pinyin import Hanzi2Pinyin
# 在此界面debug启用下面的导入库
# from pinyin import Hanzi2Pinyin


class Dynaimic_CSCDataset(Dataset):

    def __init__(self, data_path, vocab_path, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length

        # Vocab
        self.vocab_file = os.path.join(vocab_path, 'vocab.txt')
        self.tokenizer = BertTokenizer(self.vocab_file, do_lower_case=True)
        self.pinyin_vocab_file = os.path.join(vocab_path, 'vocab_pinyin.txt')
        self.pinyin_tokenier = BertTokenizer(self.pinyin_vocab_file, do_lower_case=False)
        self.hanzi2pinyin = Hanzi2Pinyin(vocab_path, map_path=vocab_path)

        file = self.data_path
        print('processing ', file)
        with open(file, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())
        # 加上小于192限制后，数据量变成了277802，原来是277804
        # 限制长度为170是因为拼音最大长度为512=170*3+2，2个特殊token为[CLS]和[SEP]
        # self.data = [line for line in self.data if len(self.pinyin_tokenier.encode(json.loads(line)['src_pinyin'])) == len(self.pinyin_tokenier.encode(json.loads(line)['tgt_pinyin']))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = json.loads(self.data[idx])
        src, tgt = example['src'], example['tgt']
        src_pinyin, tgt_pinyin = example['src_pinyin'], example['tgt_pinyin']

        input_ids = self.tokenizer.encode(src)
        label_ids = self.tokenizer.encode(tgt)
        pinyin_input_ids = self.pinyin_tokenier.encode(src_pinyin)
        pinyin_label_ids = self.pinyin_tokenier.encode(tgt_pinyin)

        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        label_ids = torch.LongTensor(label_ids)
        det_labels = torch.where(input_ids==label_ids, 0, 1)
        assert len(input_ids)==len(det_labels)
        if len(pinyin_input_ids)!=len(pinyin_label_ids):
            print(idx, src, tgt)
            return torch.LongTensor([0]), torch.LongTensor([0]), torch.LongTensor([0]), \
                torch.LongTensor([0]), torch.LongTensor([0]), torch.LongTensor([0])
        pinyin_input_ids = torch.LongTensor(pinyin_input_ids)
        pinyin_label_ids = torch.LongTensor(pinyin_label_ids)
        pinyin_det_labels = torch.where(pinyin_input_ids==pinyin_label_ids, 0, 1)

        return input_ids, label_ids, det_labels, \
                 pinyin_input_ids, pinyin_label_ids, pinyin_det_labels



class TestCSCDataset(Dataset):

    def __init__(self, data_path, vocab_path, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length

        # Vocab
        self.vocab_file = os.path.join(vocab_path, 'vocab.txt')
        self.pinyin_vocab_file = os.path.join(vocab_path, 'vocab_pinyin.txt')
        self.pinyin_tokenier = BertTokenizer(self.pinyin_vocab_file, do_lower_case=False)
        self.hanzi2pinyin = Hanzi2Pinyin(vocab_path, map_path=vocab_path)
        self.tokenizer = BertTokenizer(self.vocab_file, do_lower_case=True)

        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)
        # print(self.data[297])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert sentence to ids
        src = self.data[idx]['src']
        input_ids = self.tokenizer.encode(src)
        src_pinyin, _ = self.hanzi2pinyin.convert_sentence_to_pinyin_ids(src)
        pinyin_input_ids = self.pinyin_tokenier.encode(src_pinyin)
        # assert 
        assert len(input_ids) <= self.max_length
        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        pinyin_input_ids = torch.LongTensor(pinyin_input_ids)

        tgt = self.data[idx]['tgt']
        label_ids = self.tokenizer.encode(tgt)
        assert len(input_ids) == len(label_ids)
        tgt_pinyin, _ = self.hanzi2pinyin.convert_sentence_to_pinyin_ids(tgt)
        pinyin_label_ids = self.pinyin_tokenier.encode(tgt_pinyin)
        label_ids = torch.LongTensor(label_ids)
        pinyin_label_ids = torch.LongTensor(pinyin_label_ids)

        example_id=self.data[idx]['id']
        tokens_size=self.data[idx]['tokens_size']
        return input_ids, pinyin_input_ids, label_ids, pinyin_label_ids, example_id, src, tokens_size


class TestCSCDIMEDataset(Dataset):
    """
        处理cscd-ime的test split
    """
    def __init__(self, data_path, vocab_path, max_length=170):
        super().__init__()
        self.data_path = data_path
        self.max_length = max_length

        # Vocab
        self.vocab_file = os.path.join(vocab_path, 'vocab.txt')
        self.pinyin_vocab_file = os.path.join(vocab_path, 'vocab_pinyin.txt')
        self.pinyin_tokenier = BertTokenizer(self.pinyin_vocab_file, do_lower_case=False)
        self.hanzi2pinyin = Hanzi2Pinyin(vocab_path, map_path=vocab_path)
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file, lowercase=True)

        print('processing ', self.data_path)
        with open(self.data_path, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())
        self.data = [line.strip().lower() for line in self.data if len(line.split('\t')[1]) < self.max_length]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # convert sentence to ids
        is_error, src, tgt=self.data[idx].split('\t')
        assert len(src)==len(tgt)

        input_ids = self.tokenizer.encode(src).ids
        src_pinyin, _ = self.hanzi2pinyin.convert_sentence_to_pinyin_ids(src)
        pinyin_input_ids = self.pinyin_tokenier.encode(src_pinyin)
        # assert 
        assert len(input_ids) <= self.max_length
        # convert list to tensor
        input_ids = torch.LongTensor(input_ids)
        pinyin_input_ids = torch.LongTensor(pinyin_input_ids)

        label_ids = self.tokenizer.encode(tgt).ids
        assert len(input_ids) == len(label_ids)
        tgt_pinyin, _ = self.hanzi2pinyin.convert_sentence_to_pinyin_ids(tgt)
        pinyin_label_ids = self.pinyin_tokenier.encode(tgt_pinyin)
        label_ids = torch.LongTensor(label_ids)
        pinyin_label_ids = torch.LongTensor(pinyin_label_ids)

        # Field: example_id
        example_id="id="+str(idx)
        # Field: tokens_size
        encoded = self.tokenizer.encode(src)
        tokens = encoded.tokens[1:-1]
        tokens_size = []
        for t in tokens:
            if t == '[UNK]':
                tokens_size.append(1)
            elif t.startswith('##'):
                tokens_size.append(len(t) - 2)
            else:
                tokens_size.append(len(t))
        return input_ids, pinyin_input_ids, label_ids, pinyin_label_ids, example_id, src, tokens_size


if __name__ == '__main__':
    # 测试Dynaimic_CSCDataset
    # dataset = Dynaimic_CSCDataset(
    #             data_path="../data/train_all",
    #             vocab_path="./",
    #         )
    # print(dataset)
    # sent="她说：“我今天很高兴！”。"
    # input_ids = dataset.tokenizer.encode(sent)
    # print(input_ids)

    # item = dataset[249819]

    # train_loader = DataLoader(dataset, batch_size=4)
    # for batch in train_loader:
    #     print(batch)
    #     break

    # 测试TestCSCDataset
    dataset = TestCSCDIMEDataset(
                data_path='../../cscd-ime/data/cscd-ime/dev.tsv',
                vocab_path='./')
    print(dataset[332])