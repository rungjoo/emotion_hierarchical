# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
from torch.nn.functional import softmax
import json

from dataset import goemotion_loader
from model import EmoModel

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import pdb
import argparse, logging
from sklearn.metrics import precision_recall_fscore_support

def pdloss(batch_pred_distribution, batch_label_distribution):
    """
    batch_pred_distribution: (batch, clsNum)
    batch_label_distribution: (batch, clsNum)
    """
    batch_log_pred_distribution = torch.log(batch_pred_distribution)
    
    loss_val = 0
    for log_pred_distribution, label_distribution in zip(batch_log_pred_distribution, batch_label_distribution):
        for log_pred_prob, label_prob in zip(log_pred_distribution, label_distribution):
            loss_val -= label_prob*log_pred_prob
    return loss_val/len(batch_pred_distribution)
    
## finetune RoBETa-large
def main():
    """Dataset Loading"""
    class_type = args.class_type # 'fine_grained'
    model_type = 'bert-base-cased'
    batch_size = args.batch
    
    train_path = './dataset/train.txt'
    dev_path = './dataset/dev.txt'
    test_path = './dataset/test.txt'
    
    train_dataset = goemotion_loader(train_path, model_type)
    dev_dataset = goemotion_loader(dev_path, model_type)
    test_dataset = goemotion_loader(test_path, model_type)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    
    """logging and path"""
    save_path = f"./{class_type}"
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    log_path = os.path.join(save_path, 'train.log')
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    """Model Loading"""
    model = EmoModel(model_type, class_type)
    model = model.cuda()    
    model.train() 
    
    """Training Setting"""        
    training_epochs = args.epoch
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    """Input & Label Setting"""
    best_dev_fscore = 0
    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, data in enumerate(tqdm(train_dataloader)):
            """Prediction"""
            batch_input_ids, batch_attention_mask, fine_batch_labels, ekman_batch_labels, senti_batch_labels = data
            if class_type == 'fine_grained':
                batch_labels = fine_batch_labels
            elif class_type == 'ekman':
                batch_labels = ekman_batch_labels
            elif class_type == 'sentiment':
                batch_labels = senti_batch_labels
            batch_input_ids, batch_attention_mask, batch_labels = batch_input_ids.cuda(), batch_attention_mask.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_ids, batch_attention_mask)
            pred_distribution = softmax(pred_logits, 1)

            """Loss calculation & training"""
            loss_val = pdloss(pred_distribution, batch_labels)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        """Dev & Test evaluation"""
        model.eval()
        dev_pre, dev_rec, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader, args)
        _, _, dev_f_macro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='macro')
        _, _, dev_f_micro, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='micro')
        
        logger.info('Epoch: {}'.format(epoch))
        logger.info(f"Dev ## precision: {dev_pre}, recall: {dev_rec}, macro-F1: {dev_f_macro}, micro-F1: {dev_f_micro}")
                
        """Best Score & Model Save"""
        dev_score = sum(dev_pre)/len(dev_pre) + sum(dev_rec)/len(dev_rec) + dev_f_macro + dev_f_micro
        if dev_score > best_dev_fscore:
            best_dev_fscore = dev_score

            test_pre, test_rec, test_pred_list, test_label_list = _CalACC(model, test_dataloader, args)
            _, _, test_f_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
            _, _, test_f_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='micro')             

            _SaveModel(model, save_path)            
            logger.info(f"Test ## precision: {test_pre}, recall: {test_rec}, macro-F1: {test_f_macro}, micro-F1: {test_f_micro}")
        
        logger.info('')    
    logger.info(f"Final Test ## precision: {test_pre}, recall: {test_rec}, macro-F1: {test_f_macro}, micro-F1: {test_f_micro}")
    
def _CalACC(model, dataloader, args):
    model.eval()
    label_list = []
    pred_list = []
    
    p1_list, p2_list, p3_list = [], [], []
    r1_list, r2_list, r3_list = [], [], []
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_ids, batch_attention_mask, fine_batch_labels, ekman_batch_labels, senti_batch_labels = data
            if args.class_type == 'fine_grained':
                batch_labels = fine_batch_labels
            elif args.class_type == 'ekman':
                batch_labels = ekman_batch_labels
            elif args.class_type == 'sentiment':
                batch_labels = senti_batch_labels
            batch_input_ids, batch_attention_mask, batch_labels = batch_input_ids.cuda(), batch_attention_mask.cuda(), batch_labels.cuda()
            
            pred_logits = model(batch_input_ids, batch_attention_mask)
            pred_distribution = softmax(pred_logits, 1) # (1, clsNum)
             
            pred_logits_sort = pred_logits.sort(descending=True)
            pred_indices = pred_logits_sort.indices.tolist()[0]
            
            pred_label = pred_indices[0] # pred_logits.argmax(1).item()            
            true_labels = []
            for ind, label in enumerate(batch_labels.squeeze(0)):
                if label > 0:
                    true_labels.append(ind)
            
            """Caculation for F1"""            
            pred_list.append(pred_label)
            label_list.append(batch_labels.argmax(1).item()) # noise가 낄 수 있음. 레이블이 딱 2개인 경우 등
                
            """Calculation precision@k and recall@k"""
            p1, p2, p3 = 0, 0, 0
            r1, r2, r3 = 0, 0, 0
            
            for pred_ind in pred_indices[:1]:
                if pred_ind in true_labels:
                    p1 += 1
                    r1 += 1/len(true_labels)
                    
            for pred_ind in pred_indices[:2]:
                if pred_ind in true_labels:
                    p2 += 1/2
                    r2 += 1/len(true_labels)
                    
            for pred_ind in pred_indices[:3]:
                if pred_ind in true_labels:
                    p3 += 1/3
                    r3 += 1/len(true_labels)
            
            p1_list.append(p1)
            p2_list.append(p2)
            p3_list.append(p3)
            
            r1_list.append(r1)
            r2_list.append(r2)
            r3_list.append(r3)
            
        p1score = round(sum(p1_list)/len(p1_list)*100, 2)
        p2score = round(sum(p2_list)/len(p1_list)*100, 2)
        p3score = round(sum(p3_list)/len(p1_list)*100, 2)
        
        r1score = round(sum(r1_list)/len(p1_list)*100, 2)
        r2score = round(sum(r2_list)/len(p1_list)*100, 2)
        r3score = round(sum(r3_list)/len(p1_list)*100, 2)
        
    return [p1score, p2score, p3score], [r1score, r2score, r3score], pred_list, label_list
        
def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--batch", type=int, help = "batch_size", default = 1)
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-6) # 1e-5
    parser.add_argument( "--class_type", type=str, help = "fine_grained or ekman or sentiment", default = "fine_grained")
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    