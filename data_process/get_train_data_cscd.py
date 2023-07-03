"""
这段代码对原始数据集进行如下处理：
1. 合并sighan13,14,15和Wang271K
2. 对原始的每一句句子，对src和tgt分别获取拼音序列，如：
    src="海是由咸水构成的。"
    src_pinyin="h ai 4 sh i 4 y ou 2 x ian 2 sh ui 3 z u 3 ch eng 2 d e "
3. 得到tokenizer后的src, tgt, src_pinyin, tgt_pinyin
"""

import argparse
import random

from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import os
import json
import sys
sys.path.append('/home/haoping/Projects/chinese_spell_checking/pinyinDAE/')
from datas.pinyin import Hanzi2Pinyin



def all_train_data_to_pickle_with_pinyin(data_path, output_dir, vocab_path, max_len ):
    def _build_dataset(data_path):
        print('processing ', data_path)
        return build_dataset_with_pinyin(
        data_path=data_path,
        vocab_path=vocab_path,
        max_len=max_len
    )

    cscd_trainset = _build_dataset(data_path=os.path.join(data_path, 'train.tsv'))
    # random.shuffle(cscd_trainset)

    def write_data_to_txt(data, out_file):
        # 写入train_all, 每一行都是一个json字典，代表一个sample
        with open(out_file, 'w', encoding='utf8',) as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False)+'\n')
        print("Wrote %d total instances to %s", len(data), out_file)

    write_data_to_txt(cscd_trainset, os.path.join(output_dir, 'cscd_trainset.json'))


def build_dataset_with_pinyin(data_path, vocab_path, max_len):
    # Load Data
    data_raw = []
    with open(data_path, encoding='utf8') as f:
        data_raw = [s.split('\t') for s in f.read().splitlines()]
    print(f'#Item: {len(data_raw)} from "{data_path}"')

    # Vocab
    vocab_file = os.path.join(vocab_path, 'vocab.txt')
    tokenizer = BertWordPieceTokenizer(vocab_file, lowercase = True)
    # pinyin_vocab_file = os.path.join(vocab_path, 'pinyin_vocab.txt')
    # pinyin_tokenier = BertTokenizer(pinyin_vocab_file, do_lower_case=False)
    hanzi2pinyin = Hanzi2Pinyin(vocab_path, map_path='../datas/')

    # Data Basic
    data = []
    for item_raw in tqdm(data_raw, desc='Build Dataset'):
        # Field: is_error, src, tgt
        item = {
            'is_error': item_raw[0],
            'src': item_raw[1],
            'tgt': item_raw[2],
        }
        assert len(item['src']) == len(item['tgt'])
        data.append(item)

        # Field: tokens_size
        encoded = tokenizer.encode(item['src'])
        # tokens = encoded.tokens[1:-1]
        # tokens_size = []
        # for t in tokens:
        #     if t == '[UNK]':
        #         tokens_size.append(1)
        #     elif t.startswith('##'):
        #         tokens_size.append(len(t) - 2)
        #     else:
        #         tokens_size.append(len(t))
        # item['tokens_size'] = tokens_size

        # Field: hanzi
        item['input_ids'] = encoded.ids
        # item['label_ids'] = tokenizer.encode(item['tgt'])
        # assert len(item['input_ids']) == len(item['label_ids'])

        # Field: pinyin
        item['src_pinyin'], _ = hanzi2pinyin.convert_sentence_to_pinyin_ids(item['src'])
        item['tgt_pinyin'], _ = hanzi2pinyin.convert_sentence_to_pinyin_ids(item['tgt'])


    # Trim
    if max_len > 0:
        n_all_items = len(data)
        data = [item for item in data if len(item['input_ids']) <= max_len]
        n_filter_items = len(data)
        n_cut = n_all_items - n_filter_items
        print(f'max_len={max_len}, {n_all_items} -> {n_filter_items} ({n_cut})')

    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="../../scope/data")
    parser.add_argument('--vocab_path', default="../datas")
    parser.add_argument('--output_dir', default="../data")
    parser.add_argument('--max_len', type=int, default= 170)
    args = parser.parse_args()

    all_train_data_to_pickle_with_pinyin(
        data_path=args.data_path,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        max_len=args.max_len,
    )
"""
python get_train_data_cscd.py \
    --data_path ../../cscd-ime/data/cscd-ime/ \
    --output_dir ../data \
"""