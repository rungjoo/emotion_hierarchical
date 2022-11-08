import torch
from torch.utils.data import Dataset, DataLoader
import json
import pdb, os
import random
from transformers import BertTokenizer
    
class goemotion_loader(Dataset):
    def __init__(self, data_path, model_type):
        f = open(data_path, 'r')
        self.datalist = f.readlines()
        f.close()
            
        model_path = os.path.join('/data/project/rw/rung/model', model_type) # bert-base-cased
        self.tokenizer = BertTokenizer.from_pretrained(model_path)  
        
        f = open('../dataset/emotions.txt', 'r')
        self.fineList = f.readlines()
        self.fineList = [x.strip() for x in self.fineList]
        f.close()        
        
        with open("../dataset/ekman_mapping.json", 'r') as f:
            self.ekman_mapping_data = json.load(f)
            self.ekman_mapping_data['neutral'] = ['neutral']
            self.ekmanList = list(self.ekman_mapping_data.keys())
            self.ekman_reverse_data = {}
            for big, small_list in self.ekman_mapping_data.items():
                for small in small_list:
                    self.ekman_reverse_data[small] = big
                    
        with open("../dataset/sentiment_mapping.json", 'r') as f:
            self.senti_mapping_data = json.load(f)
            self.senti_mapping_data['neutral'] = ['neutral']
            self.sentiList = list(self.senti_mapping_data.keys())
            self.senti_reverse_data = {}
            for big, small_list in self.senti_mapping_data.items():
                for small in small_list:
                    self.senti_reverse_data[small] = big                       
        
        self.ekmanList = list(self.ekman_mapping_data.keys())
        self.sentiList = list(self.senti_mapping_data.keys())
        
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = self.datalist[idx]
        utt, labels = data.strip().split('\t')
        label_list = labels.split(',')        
        
        # fine-grained
        fine_label_list = [int(x) for x in label_list]
        
        # ekman & sentiment
        ekman_label_list, senti_label_list = [], []
        for label in label_list:
            emotion = self.fineList[int(label)]
            
            ekman_emotion = self.ekman_reverse_data[emotion]
            ind = self.ekmanList.index(ekman_emotion)
            ekman_label_list.append(ind)
            
            senti_emotion = self.senti_reverse_data[emotion]
            ind = self.sentiList.index(senti_emotion)
            senti_label_list.append(ind)          
            
        data = {}
        data['utt'] = utt
        data['fine_labels'] = fine_label_list
        data['ekman_labels'] = ekman_label_list
        data['senti_labels'] = senti_label_list
        return data
    
    def encode_truncated(self, text):
        max_length = self.tokenizer.model_max_length
        tokenized_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        truncated_tokens = tokenized_tokens[-max_length:]    

        return truncated_tokens  
    
    def padding(self, ids_list):
        max_len = 0
        for ids in ids_list:
            if len(ids) > max_len:
                max_len = len(ids)

        pad_ids = []
        for ids in ids_list:
            pad_len = max_len-len(ids)
            add_ids = [self.tokenizer.pad_token_id for _ in range(pad_len)]

            pad_ids.append(ids+add_ids)

        return torch.tensor(pad_ids)     
    
    def collate_fn(self, data):
        batch_utts = []
        fine_batch_labels, ekman_batch_labels, senti_batch_labels = [], [], []
        batch = len(data)
            
        for session in data: 
            utt = session['utt']
            fine_labels, ekman_labels, senti_labels = session['fine_labels'], session['ekman_labels'], session['senti_labels']
            
            batch_utts.append(utt)
            
            # fine
            fine_batch_label = [0 for _ in range(len(self.fineList))]
            for index in fine_labels:
                fine_batch_label[index] += 1/len(fine_labels)
            fine_batch_labels.append(fine_batch_label)
            
            # ekman
            ekman_batch_label = [0 for _ in range(len(self.ekmanList))]
            for index in ekman_labels:
                ekman_batch_label[index] += 1/len(ekman_labels)
            ekman_batch_labels.append(ekman_batch_label)
            
            # senti
            senti_batch_label = [0 for _ in range(len(self.sentiList))]
            for index in senti_labels:
                senti_batch_label[index] += 1/len(senti_labels)
            senti_batch_labels.append(senti_batch_label)            
        
        result = self.tokenizer.batch_encode_plus(batch_utts, add_special_tokens=False, padding=True, return_tensors='pt')
        batch_input_ids = result['input_ids']
        batch_attention_mask = result['attention_mask']
        
        final_input_ids = torch.cat([torch.tensor([self.tokenizer.cls_token_id for _ in range(batch)]).unsqueeze(1), batch_input_ids], 1)
        final_attention_mask = torch.cat([torch.tensor([1 for _ in range(batch)]).unsqueeze(1), batch_input_ids], 1)

        return final_input_ids, final_attention_mask, torch.tensor(fine_batch_labels), torch.tensor(ekman_batch_labels), torch.tensor(senti_batch_labels)