from datasets import load_dataset
from utils import Hanzi2Pinyin
from transformers import BertTokenizer
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'


# 数据处理参数
wiki_path = "../../../corpus/processed/wiki_sentences.txt"
vocab_path = './pinyin_vocab.txt'
pinyin2smymsd_map_path = './'
block_size = 512
test_ratio = 0.1

pinyin_tokenizer=BertTokenizer(vocab_path, do_lower_case=False)
hanzi2pinyin = Hanzi2Pinyin('./', map_path=pinyin2smymsd_map_path)
print("pinyin_tokenizer: ", pinyin_tokenizer)
print("pinyin_tokenizer.vocab_size: ", pinyin_tokenizer.vocab_size)


def tokenize_function(example):
    pinyin_ids, tokenizer_output = hanzi2pinyin.convert_sentence_to_pinyin_ids(example['text'])
    return pinyin_tokenizer(pinyin_ids)


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


dataset = load_dataset("text", data_files={"train": wiki_path})
dataset=dataset['train'].train_test_split(test_size=test_ratio, shuffle=False)
print("dataset: ", dataset)
# print(dataset['train'][:3], dataset['test'][:3])

tokenized_dataset = dataset.map(tokenize_function, 
                                        batched=False,
                                        num_proc=16,
                                        remove_columns=["text"])
print("tokenized_dataset: ", tokenized_dataset)
lm_datasets = tokenized_dataset.map(
    group_texts,
    batched=True,
    batch_size=500,
    num_proc=16,
)
print("lm_datasets: ", lm_datasets)
lm_datasets.save_to_disk("./wiki_pinyin/")