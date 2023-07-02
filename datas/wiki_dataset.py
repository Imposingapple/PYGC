import json
import os
import random
from typing import List
from functools import partial
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer

root_path = "/home/haoping/Projects/chinese_spell_checking/pinyinDAE/"
tokenizer = BertTokenizer(os.path.join(root_path, "datas/vocab.txt"), do_lower_case=True)
py_tokenizer = BertTokenizer(os.path.join(root_path, "datas/vocab_pinyin.txt"), do_lower_case=True)
vocab = tokenizer.vocab  # token2id, vocab["一"]=671
# print(vocab)
print("vocab['一']", vocab['一'])  # 671
# print(vocab["七"])
id2token={v:k for k,v in vocab.items()}
print("id2token[671]: ", id2token[671])

with open(os.path.join(root_path, "datas/hanzi2smymsd_in_confusion_set.json"), "r") as f:
    hanzi2smymsd = json.load(f)
# print("hanzi2smymsd['七']: ", hanzi2smymsd['七'])
with open(os.path.join(root_path, "datas/hanzi2confusion_set.json"), "r") as g:
    hanzi2confusionset = json.load(g)
# print("hanzi2confusionset['七']: ", hanzi2confusionset['七'])
# print(len(hanzi2confusionset))
# print(hanzi2confusionset.keys())
# print(list(hanzi2confusionset.keys())[1])


def torch_mask_tokens(input_ids, py_input_ids, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # print("input_ids: ", input_ids)
    label_ids = input_ids.clone()
    py_label_ids = py_input_ids.clone()
    batch_size, max_len = input_ids.shape
    masks = torch.where(input_ids==0, 1, 0)
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(label_ids.shape, mlm_probability)
    probability_matrix.masked_fill_(masks, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    label_ids[~masked_indices] = -100  # We only compute loss on masked tokens
    py_masked_indices = torch.zeros(py_label_ids.shape)
    for i in range(batch_size):
        for j in range(1, max_len-1):
            if masked_indices[i][j]:
                py_masked_indices[i][3*j-2]=1
                py_masked_indices[i][3*j-1]=1
                py_masked_indices[i][3*j]=1
        if masked_indices[i][0]:
            py_masked_indices[i][0]=1
        if masked_indices[i][max_len-1]:
            py_masked_indices[i][3*max_len-5]=1
    py_masked_indices=py_masked_indices.bool()
    py_label_ids[~py_masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(label_ids.shape, 0.8)).bool() & masked_indices
    # input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    # 40%的概率，替换为confusion set中的另一个字和其相应的拼音
    indices_confusionset = torch.bernoulli(torch.full(label_ids.shape, 0.5)).bool() & indices_replaced
    # 20%的概率，拼音不替换，字替换为confusion set中的一个字
    indices_confusionset_zi = torch.bernoulli(torch.full(label_ids.shape, 0.5)).bool() & indices_replaced & ~indices_confusionset
    # 20%的概率，字不替换，拼音替换为该字的confusion set中另一个字的拼音
    indices_confusionset_py = indices_replaced & ~indices_confusionset & ~indices_confusionset_zi

    for i in range(batch_size):
        for j in range(1, max_len-1):
            if indices_replaced[i][j]:
                # print("input_ids[i][j].item(): ", input_ids[i][j].item())
                ori =  id2token[input_ids[i][j].item()]
                # print("i,j: ", i,j)
                # print("ori: ", ori)
                if ori not in hanzi2confusionset.keys():
                    continue
                confusions = hanzi2confusionset[ori]
                new_ids = random.randint(0, len(confusions)-1)
                new_zi = confusions[new_ids]
                new_py = hanzi2smymsd[new_zi]
                # print("new_zi: {}, new_py: {}".format(new_zi, new_py))
                # 替换字
                if indices_confusionset[i][j] or indices_confusionset_zi[i][j]:
                    input_ids[i][j]=vocab[new_zi]
                # 替换拼音
                if indices_confusionset[i][j] or indices_confusionset_py[i][j]:
                    py_input_ids[i][3*j-2]=new_py[3]
                    py_input_ids[i][3*j-1]=new_py[4]
                    py_input_ids[i][3*j]=new_py[5]

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(label_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), label_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    for i in range(batch_size):
        for j in range(1, max_len-1):
            if indices_random[i][j]:
                new_ids = random.randint(0, len(hanzi2confusionset)-1)
                new_zi = list(hanzi2confusionset.keys())[new_ids]
                new_py = hanzi2smymsd[new_zi]
                # 替换字
                input_ids[i][j]=vocab[new_zi]
                # 替换拼音
                py_input_ids[i][3*j-2]=new_py[3]
                py_input_ids[i][3*j-1]=new_py[4]
                py_input_ids[i][3*j]=new_py[5]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return input_ids, label_ids, py_input_ids, py_label_ids


def collate_to_max_length_for_pretrain(batch: List[List[torch.Tensor]], max_len=None, fill_values: List[float] = None) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor), which shape is [seq_length]
        max_len: specify max length
        fill_values: specify filled values of each field
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
            output[0]: [batch, max_tokens_in_batch]     input_ids
            output[1]: [batch, max_tokens_in_batch]     label_ids
            output[2]: [batch, max_tokens_in_batch]     det_labels
            output[3]: [batch, max_pinyintokens_in_batch]   pinyin_input_ids
            output[4]: [batch, max_pinyintokens_in_batch]   pinyin_label_ids
            output[5]: [batch, max_pinyintokens_in_batch]   pinyin_det_labels
    """

    lengths = np.array([[len(field_data) for field_data in sample[:2]] for sample in batch])
    batch_size, num_fields = lengths.shape # [batch, num_fields_to_pad]
    fill_values = fill_values or [0.0] * num_fields
    
    max_lengths = lengths.max(axis=0) # [num_fields]
    if max_len:
        assert max_lengths.max() <= max_len
        max_lengths = np.ones_like(max_lengths) * max_len
    max_lengths

    output = [torch.full((batch_size, max_lengths[field_idx]),
                         fill_value=fill_values[field_idx],
                         dtype=batch[0][field_idx].dtype)
              for field_idx in range(num_fields)]
    for sample_idx in range(batch_size):
        for field_idx in range(num_fields):
            # seq_length
            data = batch[sample_idx][field_idx]
            output[field_idx][sample_idx][: data.shape[0]] = data
    input_ids_pad, py_input_ids_pad = output

    # 依据confusion set，从真实数据构造预训练的数据和标签
    return torch_mask_tokens(input_ids_pad, py_input_ids_pad)


class Wiki_Dataset(Dataset):

    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path

        file = self.data_path
        print('processing ', file)
        with open(file, 'r' ,encoding='utf8') as f:
            self.data = list(f.readlines())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx])
        #  print(type(item))
        return torch.LongTensor(item['input_ids']), torch.LongTensor(item['py_input_ids']), \
                item['offsets'], item['src'], item['src_pinyin']


if __name__=='__main__':
    # arguments
    batch_size=4

    wiki_dataset = Wiki_Dataset("../data/wiki_with_pyids_50000.txt")
    # wiki_dataset = Wiki_Dataset("../data/wiki_with_pyids.txt")
    batch = [wiki_dataset[0], wiki_dataset[1]]
    # print(batch)

    # dataloader = DataLoader(
    #         dataset=wiki_dataset,
    #         batch_size=batch_size,
    #         shuffle=False,
    #         num_workers=8,
    #         collate_fn=partial(collate_to_max_length_for_pretrain, fill_values=[0, 0]),
    #         drop_last=False,
    #     )

    # for batch in dataloader:
    #     print(batch)
    #     quit()
    collate_to_max_length_for_pretrain(batch)
    