import opencc
import os
import json
from transformers import BertTokenizer

tokenizer = BertTokenizer("../datas/vocab.txt", do_lower_case=True)
vocab = tokenizer.vocab.keys()
print(type(vocab))
converter = opencc.OpenCC('t2s.json')

confusionset_path = "../../data/sighan_original/SIGHAN2013/ConfusionSet/"
file1 = os.path.join(confusionset_path, "Bakeoff2013_CharacterSet_SimilarPronunciation.txt")
file2 = os.path.join(confusionset_path, "Bakeoff2013_CharacterSet_SimilarShape.txt")
confusion_set = {}
with open(file1, 'r') as f:
    lines = f.readlines()[1:]
    # print(lines[:10])
    for line in lines:
        line = line.strip().split('\t')
        key = converter.convert(line[0])
        if key not in vocab:
            continue
        confusions = converter.convert(''.join(line[1:]))
        value=[]
        # print(key, confusions)
        for c in confusions:
            if c in vocab:
                value.append(c)
            else:
                # print(c)
                pass
        confusion_set[key]=value
    # print(confusion_set)

with open(file2, 'r') as f:
    lines = f.readlines()[1:]
    # print(lines[:10])
    for line in lines:
        line = line.strip().split(',')
        key = converter.convert(line[0])
        if key not in vocab:
            # print("key: ", key)
            continue
        confusions = converter.convert(line[1])
        value=[]
        # print(key, confusions)
        for c in confusions:
            if c in vocab:
                value.append(c)
            else:
                # print(c)
                pass
        confusion_set[key]+=value  # 直接附在后面，这样的话声音和形都相近的字出现了两次，之后程序更容易选到

# print(confusion_set)
print(len(confusion_set.keys()))
print(confusion_set["一"])
print(confusion_set["奇"])
print(confusion_set["妈"])
print(confusion_set["我"])

with open("../datas/hanzi2confusion_set.json", "w") as g:
    json.dump(confusion_set, g, ensure_ascii=False)