"""
    这份代码得到两个json文件：
1. confusion set出现的汉字对应的拼音
2. confusion set出现的汉字的相似字（在confusion set中）对应的拼音
"""

import json
from pypinyin import pinyin, Style
from transformers import BertTokenizer
import sys
sys.path.append('/home/haoping/Projects/chinese_spell_checking/pinyinDAE/')
from datas.pinyin import Pinyin

pytool = Pinyin()
pinyin_tokenizer = BertTokenizer("../datas/vocab_pinyin.txt", do_lower_case=False)
pinyin_vocab = pinyin_tokenizer.vocab

# 对confusion set中所有字获取sm，ym，sd
with open("../datas/hanzi2confusion_set.json", "r") as f, \
    open("../datas/hanzi2smymsd_in_confusion_set.json", "w") as g:
    d = json.load(f)
    # print(d)
    hanzi2pinyin={}
    for key, value in d.items():
        candidates = [key] + value
        # print(candidates)
        for c in candidates:
            if c in hanzi2pinyin.keys():
                continue
            # 获得每个汉字和其对应的smymsd（可能有多个读法）
            pinyin_list = pinyin(c, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])[0]
            # print(pinyin_list)
            sm,ym,sd=pytool.get_sm_ym_sd(pinyin_list[0])
            smymsd_list=[sm,ym,sd]
            # print(key, smymsd_list)
            smymsd_ids_list=[pinyin_vocab[sm], pinyin_vocab[ym], pinyin_vocab[sd]]
            hanzi2pinyin[c]=smymsd_list+smymsd_ids_list
    print(len(hanzi2pinyin.keys()))  # 只统计key的smymsd是4749
    json.dump(hanzi2pinyin, g, ensure_ascii=False)

        