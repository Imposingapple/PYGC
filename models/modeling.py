import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPredictionHeadTransform
from transformers import AutoModel, BertModel, AutoModelForMaskedLM
from tokenizers import BertWordPieceTokenizer
import os
from torch.nn.parameter import Parameter


class HanziModel(nn.Module):
    def __init__(self, vocab_size, pretrained_lm, is_multitask):
        super(HanziModel, self).__init__()
        self.vocab_size = vocab_size
        self.is_multitask = is_multitask
        checkpoint = pretrained_lm
        config = AutoConfig.from_pretrained(checkpoint)
        vocab_path = "./datas/"
        vocab_file = os.path.join(vocab_path, 'vocab.txt')
        self.tokenizer = BertWordPieceTokenizer(vocab_file, lowercase = True)

        ### BertModel
        ## method 1 用automodel，解码层和BertPredictionHeadTransform层不初始化
        # # 1. 先加载pretrained BertModel
        # self.bert=AutoModel.from_pretrained(checkpoint)
        # # print("self.bert before: \n", self.bert)
        # # 2. 替换embeddings，使得词表大小和tokenizer的词表大小一致
        # config.vocab_size=self.vocab_size
        # # # print(config)
        # # embeddings = BertEmbeddings(config)
        # # self.bert.embeddings = embeddings
        # # # print("self.bert now: \n", self.bert)

        # self.transform = BertPredictionHeadTransform(config)
        # self.dropout = nn.Dropout(0.1, inplace=False)
        # self.decoder = nn.Linear(config.hidden_size, self.vocab_size, bias=True)
        # self.cls = nn.Linear(config.hidden_size, 2, bias=True)
        
        ## method2 用AutoModelForMaskedLM
        self.model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        self.bert = self.model.bert
        self.cls = self.model.cls
        # 复制原张量，不共享梯度
        # self.cls.predictions.decoder.weight = Parameter(self.cls.predictions.decoder.weight.clone().detach().requires_grad_(True))
        if self.is_multitask:
            self.det_transform = BertPredictionHeadTransform(config)
            self.detector = nn.Linear(config.hidden_size, 2, bias=True)

        # 融合pinyin language model的hidden embedding的Linear层
        self.pinyin_map = nn.Linear(config.hidden_size, config.hidden_size)
        self.map_fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    
    def forward(self, input_ids, attention_mask, pinyin_hidden_states):
        bert_outputs = self.bert(input_ids=input_ids,
                                 attention_mask=attention_mask)
        hanzi_hidden_states = bert_outputs['last_hidden_state']

        # 选取声母、韵母、音调对应的hidden
        max_len=hanzi_hidden_states.shape[1]
        pinyin_len=pinyin_hidden_states.shape[1]
        if pinyin_len!=3*max_len-4:
            print("not equal length!!!!!!")
            print("input_ids: ", input_ids)
            print("input_ids.shape: ", input_ids.shape)
            print("pinyin_len, max_len: ", pinyin_len, max_len)
            for i in range(input_ids.shape[0]):
                print(self.tokenizer.decode(input_ids[i].tolist()))
        device=hanzi_hidden_states.device
        sm_idx=[0]+[3*i+1 for i in range(max_len-2)]+[3*max_len-5]
        ym_idx=[0]+[3*i+2 for i in range(max_len-2)]+[3*max_len-5]
        sd_idx=[0]+[3*i+3 for i in range(max_len-2)]+[3*max_len-5]
        sm_hidden=torch.index_select(pinyin_hidden_states, 1, torch.tensor(sm_idx).to(device)) # [batch, max_seq, 768]
        ym_hidden=torch.index_select(pinyin_hidden_states, 1, torch.tensor(ym_idx).to(device))
        sd_hidden=torch.index_select(pinyin_hidden_states, 1, torch.tensor(sd_idx).to(device))
        concat_pinyin_hidden=torch.cat((sm_hidden, ym_hidden, sd_hidden), 2)  # [batch, max_seq, 768*3]
        new_pinyin_hidden=self.pinyin_map(concat_pinyin_hidden)
        # 将变换后的pinyin_hidden和hanzi_hidden融合
        concat_embeddings = torch.cat((hanzi_hidden_states, new_pinyin_hidden), 2)
        hidden_states = self.map_fc(concat_embeddings)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        ## method 1
        # x = self.transform(hidden_states)
        # x = self.dropout(x)
        # cor_out, det_out = self.decoder(x), self.cls(x)
        ## method 2
        cor_out = self.cls(hidden_states)
        if not self.is_multitask:
            return cor_out, None
        else:
            det_hid = self.det_transform(hidden_states)
            det_out = self.detector(det_hid)
            return cor_out, det_out


class PinyinModel(nn.Module):
    def __init__(self, vocab_size, pretrained_pinyin_lm_file, is_multitask):
        super(PinyinModel, self).__init__()
        self.vocab_size = vocab_size
        self.is_multitask = is_multitask
        checkpoint = pretrained_pinyin_lm_file
        self.config = AutoConfig.from_pretrained(checkpoint)

        ### ElectraModel
        ## method 1
        # self.electra = AutoModel.from_pretrained(checkpoint)
        # self.transform = BertPredictionHeadTransform(self.config)
        # self.dropout = nn.Dropout(0.1, inplace=False)
        # self.decoder = nn.Linear(self.config.hidden_size, self.vocab_size, bias=True)
        # self.cls = nn.Linear(self.config.hidden_size, 2)

        ## method 2
        self.model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        self.electra = self.model.electra
        self.transform = self.model.generator_predictions
        self.cls = self.model.generator_lm_head
        if self.is_multitask:
            self.detector = nn.Linear(self.config.embedding_size, 2, bias=True)
    
    def forward(self, pinyin_input_ids, pinyin_attention_mask):
        bert_outputs = self.electra(input_ids=pinyin_input_ids,
                                    attention_mask=pinyin_attention_mask)
        hidden_states = bert_outputs['last_hidden_state']

        ## method 1
        # x = self.transform(hidden_states)
        # x = self.dropout(x)
        # cor_out, det_out = self.decoder(x), self.cls(x)
        # return cor_out, det_out

        ## method 2
        x = self.transform(hidden_states)
        cor_out = self.cls(x)
        if not self.is_multitask:
            return cor_out, None, hidden_states
        else:
            det_out = self.detector(x)
            return cor_out, det_out, hidden_states


class PYGC(nn.Module):
    def __init__(self, vocab_size, pretrained_hanzi_lm, 
                    pinyin_vocab_size, pretrained_pinyin_lm_file, is_multitask):
        super(PYGC, self).__init__()
        self.hanzi_model = HanziModel(vocab_size, pretrained_hanzi_lm, is_multitask)
        self.pinyin_model = PinyinModel(pinyin_vocab_size, pretrained_pinyin_lm_file, is_multitask)
    
    def forward(self, input_ids, attention_mask,
                pinyin_input_ids, pinyin_attention_mask):
        pinyin_cor_out, pinyin_det_out, pinyin_hidden_states = self.pinyin_model(
            pinyin_input_ids,
            pinyin_attention_mask
        )
        cor_out, det_out = self.hanzi_model(
            input_ids,
            attention_mask,
            pinyin_hidden_states
        )
        
        return cor_out, det_out, pinyin_cor_out, pinyin_det_out


   
if __name__ ==  '__main__':
    import os
    from transformers import BertTokenizer
    from tokenizers import BertWordPieceTokenizer
    import torch
    # HanziModel
    # checkpoint="bert-base-chinese"
    # model=AutoModel.from_pretrained(checkpoint)
    # print(model)
    # print(model.embeddings.word_embeddings.weight.data[1000, :20])
    # config = AutoConfig.from_pretrained(checkpoint)
    # config.vocab_size=23236
    # print(config)
    # embeddings = BertEmbeddings(config)
    # print(embeddings)
    # model.embeddings = embeddings
    # print(model)

    # PinyinModel
    vocab_path = "../datas/"
    vocab_file = os.path.join(vocab_path, 'vocab.txt')
    vocab_size = len(open(vocab_file, 'r').readlines())
    tokenizer = BertWordPieceTokenizer(vocab_file, lowercase = True)
    sent = "她说：“今天我很高兴。”"
    encoded = tokenizer.encode(sent)
    print("encoded.ids: ", encoded.ids)
    # input_ids=encoded.ids
    input_ids=[ 101, 4294, 4905, 6956, 7339, 2175, 6230, 2772, 1355, 4385,  800,  812,
        3198, 1377,  809, 1406, 6435, 1092, 3175, 6822, 6121, 4958, 6159,  511,
         102,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
           0,    0,    0]
    sent=tokenizer.decode(input_ids)
    print(input_ids)
    print(sent)

