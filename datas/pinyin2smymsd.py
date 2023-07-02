"""
    这份代码将pypinyin得到的拼音转化为声母、韵母、声调
    比如：'ling2'-> ['l','ing','2]
    保存成json格式
"""
import json
from utils import Pinyin

pinyin_converter=Pinyin()

with open('../../scope/FPT/config/pinyin2tensor.json', 'r', encoding='utf8') as f, \
        open('./pinyin2smymsd.json', 'w', encoding='utf8') as g:
    json_dict = json.load(f)
    # print(json_dict)
    pinyins=json_dict.keys()

    pinyin2smymsd={}
    for pinyin in pinyins:
        # print(pinyin)
        if pinyin[-1] not in '12345':  # 原本f中的json不包含轻声'5'，直接给不包含轻声的加上'5'
            pinyin+='5'
        sm, ym, sd = pinyin_converter.get_sm_ym_sd(pinyin)
        pinyin2smymsd[pinyin]=[sm, ym, sd]
    json.dump(pinyin2smymsd, g)

    