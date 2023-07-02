import argparse
import random
import os
import json

from tqdm import tqdm
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer
from pypinyin import pinyin, Style


class Pinyin(object):
    """
        Pinyin模块负责将拼音准换成声母韵母声调
    """
    def __init__(self):
        super(Pinyin, self).__init__()
        self.shengmu = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
        self.yunmu = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie', 'in', 'ing', 'iong', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ue', 'ui', 'un', 'uo', 'v', 've']
        self.shengdiao= ['1', '2', '3', '4', '5']
        # 构建词表
        self.id2token=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'notChinese', 'noSm']
        # 在词表中添加声母韵母声调
        for sm in self.shengmu:
            self.id2token.append(sm)
        for ym in self.yunmu:
            self.id2token.append(ym)
        for sd in self.shengdiao:
            self.id2token.append(sd)
        # print(self.id2token)
        self.vocab_size=len(self.id2token)

    def write_to_vocab_file(self, path='./'):
        """
            将词表作为vocab.txt写入path
        """
        vocab_file=os.path.join(path, 'vocab_pinyin.txt')
        with open(vocab_file, 'w') as f:
            for t in self.id2token:
                f.write(t+'\n')

    def get_sm_ym_sd(self, pinyin):
        if pinyin=="n2":
            return ['e','n','2']
        s=pinyin
        assert isinstance(s, str),'input of function get_sm_ym_sd is not string'
        sm, ym, sd = None, None, None
        if not s[-1] in '12345':
            s+='5'
        
        for c in self.shengmu:
            if s.startswith(c):
                sm = c
                break
        if sm == None:
            ym = s[:-1]
        else:
            ym = s[len(sm):-1]
        sd = s[-1]

        assert ym is not None
        assert sd is not None
        if sm is None:
            sm = 'noSm'
        return sm, ym, sd


class Hanzi2Pinyin():
    """
        Hanzi2Pinyin模块负责将汉字序列转换成拼音序列
    """
    def __init__(self, chinese_bert_path, map_path='./', max_length: int = 512):
        self.vocab_file = os.path.join(chinese_bert_path, 'vocab.txt')
        # self.config_path = os.path.join(chinese_bert_path, 'config')
        self.max_length = max_length
        # self.tokenizer = BertTokenizer(self.vocab_file, do_lower_case=True)
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)
        self.pinyin2smymsd_converter = Pinyin()

        # load pinyin map dict
        # {"ling2": ["l", "ing", "2"], ..., "qiu4": ["q", "iu", "4"]}
        with open(os.path.join(map_path, 'pinyin2smymsd.json'), encoding='utf8') as f:
            self.pinyin2smymsd = json.load(f)

    def convert_sentence_to_pinyin_ids(self, sentence: str):
        """
            示例：以输入为“第20天，我很高兴啊！”为例
        """
        # get pinyin of a sentence
        # TONE3：声调风格3，即拼音声调在各个拼音之后，用数字 [1-4] 进行表示。如： 中国 -> zhong1 guo2
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        # print("pinyin_list: ", pinyin_list)
        # pinyin_list: [['di4'], ['not chinese'], ['not chinese'], ['tian1'], ['not chinese'], 
        #               ['wo3'], ['hen3'], ['gao1'], ['xing4'], ['not chinese']]
        
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2smymsd:
                # 处理pypinyin的特殊情况
                if pinyin_string=="n2": # 嗯
                    pinyin_locs[index] = ['noSm','en','2']
                else:
                    pinyin_locs[index] = self.pinyin2smymsd[pinyin_string]
            else:
                # print("pinyin_string: ", pinyin_string)
                sm, ym, sd = self.pinyin2smymsd_converter.get_sm_ym_sd(pinyin_string)
                pinyin_locs[index] = [sm, ym, sd]
        # print("pinyin_locs: ", pinyin_locs)

        # find chinese character location, and generate pinyin ids
        tokenizer_output = self.tokenizer.encode(sentence)
        # print("tokenizer_output.tokens: ", tokenizer_output.tokens)
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset==(0,0):  # 去除开头的[cls]和末尾的[seq] token
                continue
            if offset[1] - offset[0] != 1:
                pinyin_ids.append(['notChinese'] * 3)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append(['notChinese'] * 3)
        # print("pinyin_ids:", pinyin_ids)
        # pinyin_ids:[['d', 'i', '4'], ['notChinese', 'notChinese', 'notChinese'], ['t', 'ian', '1'], 
        # ['notChinese', 'notChinese', 'notChinese'], ['w', 'o', '3'], ['h', 'en', '3'], ['g', 'ao', '1'], 
        # ['x', 'ing', '4'], ['noSm', 'a', '5'], ['notChinese', 'notChinese', 'notChinese']]

        # 去除中间的空格
        pinyin_seq=[]
        for item in pinyin_ids:
            pinyin_seq.append(' '.join(item))
        return ' '.join(pinyin_seq), tokenizer_output



if __name__=='__main__':
    # pho_convertor = Pinyin()
    # print(pho_convertor.get_sm_ym_sd_labels('a1'),type(pho_convertor.get_sm_ym_sd_labels('a1')))

    # pho_convertor=Pinyin()
    # pho_convertor.write_to_vocab_file()

    # sent = "第20天，我很高兴啊！"
    sent = "嗯"
    hanzi2pinyin = Hanzi2Pinyin("./")
    print(hanzi2pinyin.convert_sentence_to_pinyin_ids(sent))