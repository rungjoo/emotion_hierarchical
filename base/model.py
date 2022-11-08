from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import os
import json

class EmoModel(nn.Module):
    def __init__(self, model_type, class_type):
        super(EmoModel, self).__init__()        
        
        model_path = os.path.join('/data/project/rw/rung/model', model_type)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        
        if class_type == 'fine_grained':
            f = open('./dataset/emotions.txt', 'r')
            emoList = f.readlines()
            f.close()
            self.clsNum = len(emoList)
        else:
            with open(f"./dataset/{class_type}_mapping.json", 'r') as f:
                mapping_data = json.load(f)
            mapping_data['neutral'] = ['neutral']
            self.clsNum = len(mapping_data.keys())
        
        self.Wc = nn.Linear(self.model.config.hidden_size, self.clsNum) # for classification

    def forward(self, batch_input_ids, batch_attention_mask):
        """
            input_tokens: (batch, len)
        """
        hidden_outs = self.model(batch_input_ids, attention_mask=batch_attention_mask)['last_hidden_state'] # [B, L, 768]
        pred_outs = self.Wc(hidden_outs) # (B, L, C)
        cls_outs = pred_outs[:,0,:] # (B, C)
        return cls_outs