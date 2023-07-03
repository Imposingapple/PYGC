"""
这段代码对wiki_dataset进行操作：
    对原始的每一句句子，获取src和对应的拼音序列src_pinyin，以及对应的ids，如：
    src:"她说：“我高兴啊”。"
    src_pinyin:"t a 1 sh uo 1 notChinese notChinese notChinese notChinese notChinese notChinese w o 3 g ao 1 x ing 4 noSm a 5 notChinese notChinese notChinese notChinese notChinese notChinese"
    input_ids:"[101, 1961, 6432, 8038, 1, 2769, 7770, 1069, 1557, 2, 511, 102]"
    py_input_ids:"[2, 15, 30, 64, 9, 61, 64, 5, 5, 5, 5, 5, 5, 29, ...]"
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


def all_train_data_to_pickle_with_pinyin(data_file, output_dir, vocab_path, max_len):
    def _build_dataset(data_file):
        print('processing ', data_file)
        return build_dataset_with_pinyin(
        data_file=data_file,
        vocab_path=vocab_path,
        max_len=max_len
    )
    wiki_dataset = _build_dataset(data_file=data_file)
    # random.shuffle(wiki_dataset)

    def write_data_to_txt(data, out_file):
        # 写入wiki_with_pyids_50000.txt, 每一行都是一个json字典，代表一个sample
        with open(out_file, 'w', encoding='utf8') as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False)+'\n')
        print("Wrote %d total instances to %s", len(data), out_file)

    write_data_to_txt(wiki_dataset, os.path.join(output_dir, 'wiki_with_pyids_50000.txt'))


def build_dataset_with_pinyin(data_file, vocab_path, max_len):
    # Load Data
    data_raw = []
    with open(data_file, encoding='utf8') as f:
        data_raw = [s.strip() for s in f.read().splitlines()]
    print(f'#Item: {len(data_raw)} from "{data_file}"')

    # Vocab
    vocab_file = os.path.join(vocab_path, 'vocab.txt')
    tokenizer = BertWordPieceTokenizer(vocab_file, lowercase = True)
    pinyin_vocab_file = os.path.join(vocab_path, 'vocab_pinyin.txt')
    pinyin_tokenier = BertTokenizer(pinyin_vocab_file, do_lower_case=False)
    hanzi2pinyin = Hanzi2Pinyin(vocab_path, map_path='../datas/')

    # Data Basic
    data = []
    for idx, src in enumerate(tqdm(data_raw, desc='Build Dataset')):
        item = {
            'src': src.strip(),
        }

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

        item['input_ids'] = encoded.ids
        item['offsets'] = encoded.offsets
        item['src_pinyin'], _ = hanzi2pinyin.convert_sentence_to_pinyin_ids(item['src'])
        item['py_input_ids'] = pinyin_tokenier.encode(item['src_pinyin'])
        assert len(item['input_ids'])==len(item['offsets'])

        if len(item['py_input_ids'])==3*len(item['input_ids'])-4:
            data.append(item)
        else:
            print(idx, src)

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
    parser.add_argument('--data_file', default="../../../../Corpus/wiki_zh_2019/processed/wiki_sentences_50000.txt")
    parser.add_argument('--vocab_path', default="../datas")
    parser.add_argument('--output_dir', default="../data")
    parser.add_argument('--max_len', type=int, default= 170)
    args = parser.parse_args()

    # 测试单样本
    # src = "而英文词汇Ocean即“洋”则可追溯到古希腊语中表示环绕大地的大洋神Ὠκεανός，一般汉译为“俄刻阿诺斯”。"
    # src="2015年在浙江卫视的《爸爸回来了第二季》中，杜江带着儿子嗯哼参加节目，节目的热播让更多人认识杜江。"
    # src="她说：“我高兴 了20天”。"
    # # src="道德巷与海边新街之间的一段火船头街原属海边新街的一部份。"
    # vocab_file = os.path.join(args.vocab_path, 'vocab.txt')
    # tokenizer = BertWordPieceTokenizer(vocab_file, lowercase = True)
    # pinyin_vocab_file = os.path.join(args.vocab_path, 'vocab_pinyin.txt')
    # pinyin_tokenier = BertTokenizer(pinyin_vocab_file, do_lower_case=False)
    # hanzi2pinyin = Hanzi2Pinyin(args.vocab_path, map_path='../datas/')

    # encoded = tokenizer.encode(src)
    # input_ids = encoded.ids
    # src_pinyin, _ = hanzi2pinyin.convert_sentence_to_pinyin_ids(src)
    # py_input_ids = pinyin_tokenier.encode(src_pinyin)
    # print("testing done")

    all_train_data_to_pickle_with_pinyin(
        data_file=args.data_file,
        output_dir=args.output_dir,
        vocab_path=args.vocab_path,
        max_len=args.max_len,
    )

    print("done")
"""
python data_process/get_train_data.py \
    --data_path data \
    --output_dir data
"""
