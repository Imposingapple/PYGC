from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer

sent = "她说：“今天我很高兴！”。"
vocab_file = "./vocab.txt"
tokenizer1 = BertTokenizer(vocab_file, do_lower_case=True)
tokenizer2 = BertWordPieceTokenizer(vocab_file)

encoded1 = tokenizer1.encode(sent)
encoded2 = tokenizer2.encode(sent)
encoded1