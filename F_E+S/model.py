from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import os
import json

class EmoModel(nn.Module):
    def __init__(self, model_type):
        super(EmoModel, self).__init__()        
        
        model_path = os.path.join('/data/project/rw/rung/model', model_type)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        
        f = open('../dataset/emotions.txt', 'r')
        emoList = f.readlines()
        f.close()
        self.fclsNum = len(emoList)

        with open(f"../dataset/ekman_mapping.json", 'r') as f:
            mapping_data = json.load(f)
        mapping_data['neutral'] = ['neutral']
        self.eclsNum = len(mapping_data.keys())
        
        with open(f"../dataset/sentiment_mapping.json", 'r') as f:
            mapping_data = json.load(f)
        mapping_data['neutral'] = ['neutral']
        self.sclsNum = len(mapping_data.keys())        
        
        self.Wf = nn.Linear(self.model.config.hidden_size, self.fclsNum) # for classification
        self.We = nn.Linear(self.fclsNum, self.eclsNum) # for classification
        self.Ws = nn.Linear(self.fclsNum, self.sclsNum) # for classification

    def forward(self, batch_input_ids, batch_attention_mask):
        """
            input_tokens: (batch, len)
        """
        hidden_outs = self.model(batch_input_ids, attention_mask=batch_attention_mask)['last_hidden_state'] # [B, L, 768]
        fpred_outs = self.Wf(hidden_outs) # (B, L, C)
        epred_outs = self.We(fpred_outs) # (B, L, C)
        spred_outs = self.Ws(fpred_outs) # (B, L, C)
        
        fcls_outs = fpred_outs[:,0,:] # (B, C)
        ecls_outs = epred_outs[:,0,:] # (B, C)
        scls_outs = spred_outs[:,0,:] # (B, C)
        return fcls_outs, ecls_outs, scls_outs