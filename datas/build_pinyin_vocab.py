from pinyin import Hanzi2Pinyin
from transformers import BertTokenizer

# special_tokens={'unk_token': '[UNK]', 'sep_token': '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}
pinyin_tokenizer=BertTokenizer('pinyin_vocab.txt', do_lower_case=False)
print("pinyin_tokenizer: ", pinyin_tokenizer)

sent1="第20天，我很高兴啊！"
# sent="PWTC是布特拉世界贸易中心（Putra World Trade Center）的缩写。"
# sent="在太平洋这些可能覆盖多达30%的海海海底。"
sent2="大海对地球的气候起到缓冲作用，且在水循环、 碳循环，以及氮循环中扮演着重要的角色。"

hanzi2pinyin = Hanzi2Pinyin('../../scope/FPT')
pinyin_ids1, tokenizer_output1 = hanzi2pinyin.convert_sentence_to_pinyin_ids(sent1)
pinyin_ids2, tokenizer_output2 = hanzi2pinyin.convert_sentence_to_pinyin_ids(sent2)
print(pinyin_ids1)
print(pinyin_ids2)

encoded = pinyin_tokenizer([pinyin_ids1, pinyin_ids2], padding='max_length', max_length=100)
print(encoded)
print(type(encoded.input_ids))

