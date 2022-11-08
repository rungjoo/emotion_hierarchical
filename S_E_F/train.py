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
    
    train_path = '../dataset/train.txt'
    dev_path = '../dataset/dev.txt'
    test_path = '../dataset/test.txt'
    
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
    model = EmoModel(model_type)
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
            
            batch_input_ids, batch_attention_mask = batch_input_ids.cuda(), batch_attention_mask.cuda()
            fine_batch_labels, ekman_batch_labels, senti_batch_labels = fine_batch_labels.cuda(), ekman_batch_labels.cuda(), senti_batch_labels.cuda()
            
            fpred_logits, epred_logits, spred_logits = model(batch_input_ids, batch_attention_mask)
            fpred_distribution = softmax(fpred_logits, 1)
            epred_distribution = softmax(epred_logits, 1)
            spred_distribution = softmax(spred_logits, 1)

            """Loss calculation & training"""
            floss_val = pdloss(fpred_distribution, fine_batch_labels)
            eloss_val = pdloss(epred_distribution, ekman_batch_labels)
            sloss_val = pdloss(spred_distribution, senti_batch_labels)
            loss_val = floss_val + eloss_val + sloss_val
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        """Dev & Test evaluation"""
        model.eval()
        result = _CalACC(model, dev_dataloader, args)
        dev_fine_grained_result = result['fine_grained']
        dev_ekman_grained_result = result['ekman']
        dev_sentiment_grained_result = result['sentiment']
    
        logger.info('Epoch: {}'.format(epoch))
        fdev_pre, fdev_rec, fdev_f_macro, fdev_f_micro = finalscore(dev_fine_grained_result)
        logger.info(f"(Dev) Fine-grained ## precision: {fdev_pre}, recall: {fdev_rec}, macro-F1: {round(fdev_f_macro*100,2)}, micro-F1: {round(fdev_f_micro*100,2)}")
        
        edev_pre, edev_rec, edev_f_macro, edev_f_micro = finalscore(dev_ekman_grained_result)
        logger.info(f"(Dev) Ekman ## precision: {edev_pre}, recall: {edev_rec}, macro-F1: {round(edev_f_macro*100,2)}, micro-F1: {round(edev_f_micro*100,2)}")
        
        sdev_pre, sdev_rec, sdev_f_macro, sdev_f_micro = finalscore(dev_sentiment_grained_result)
        logger.info(f"(Dev) Sentiment ## precision: {sdev_pre}, recall: {sdev_rec}, macro-F1: {round(sdev_f_macro*100,2)}, micro-F1: {round(sdev_f_micro*100,2)}")
                
        """Best Score & Model Save"""
        dev_score = sum(fdev_pre)/len(fdev_pre) + sum(fdev_rec)/len(fdev_rec) + fdev_f_macro + fdev_f_micro\
                    + sum(edev_pre)/len(edev_pre) + sum(edev_rec)/len(edev_rec) + edev_f_macro + edev_f_micro\
                    + sum(sdev_pre)/len(sdev_pre) + sum(sdev_rec)/len(sdev_rec) + sdev_f_macro + sdev_f_micro
        if dev_score > best_dev_fscore:
            best_dev_fscore = dev_score

            result = _CalACC(model, test_dataloader, args)
            test_fine_grained_result = result['fine_grained']
            test_ekman_grained_result = result['ekman']
            test_sentiment_grained_result = result['sentiment']

            fpre, frec, fmacro, fmicro = finalscore(test_fine_grained_result)
            logger.info(f"(Test) Fine-grained ## precision: {fpre}, recall: {frec}, macro-F1: {round(fmacro*100,2)}, micro-F1: {round(fmicro*100,2)}")
            
            epre, erec, emacro, emicro = finalscore(test_ekman_grained_result)
            logger.info(f"(Test) Ekman ## precision: {epre}, recall: {erec}, macro-F1: {round(emacro*100,2)}, micro-F1: {round(emicro*100,2)}")
            
            spre, srec, smacro, smicro = finalscore(test_sentiment_grained_result)
            logger.info(f"(Test) Sentiment ## precision: {spre}, recall: {srec}, macro-F1: {round(smacro*100,2)}, micro-F1: {round(smicro*100,2)}")

            _SaveModel(model, save_path)
        
        logger.info('')    
    logger.info(f"(Final Test) Fine-grained ## precision: {fpre}, recall: {frec}, macro-F1: {round(fmacro*100,2)}, micro-F1: {round(fmicro*100,2)}")
    logger.info(f"(Final Test) Ekman ## precision: {epre}, recall: {erec}, macro-F1: {round(emacro*100,2)}, micro-F1: {round(emicro*100,2)}")
    logger.info(f"(Final Test) Sentiment ## precision: {spre}, recall: {srec}, macro-F1: {round(smacro*100,2)}, micro-F1: {round(smicro*100,2)}")
    
def finalscore(result):
    test_pre = result['p']
    test_rec = result['r']
    test_pred_list = result['pred_list']
    test_label_list = result['label_list']

    _, _, test_f_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
    _, _, test_f_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='micro')   
    
    return test_pre, test_rec, test_f_macro, test_f_micro    
    
def _CalACC(model, dataloader, args):
    model.eval()
    fpred_list, flabel_list = [], []
    epred_list, elabel_list = [], []
    spred_list, slabel_list = [], []
    
    fp1_list, fp2_list, fp3_list = [], [], []
    ep1_list, ep2_list, ep3_list = [], [], []
    sp1_list, sp2_list, sp3_list = [], [], []
    
    fr1_list, fr2_list, fr3_list = [], [], []
    er1_list, er2_list, er3_list = [], [], []
    sr1_list, sr2_list, sr3_list = [], [], []
    # label arragne
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_ids, batch_attention_mask, fine_batch_labels, ekman_batch_labels, senti_batch_labels = data
            
            batch_input_ids, batch_attention_mask = batch_input_ids.cuda(), batch_attention_mask.cuda()
            fine_batch_labels, ekman_batch_labels, senti_batch_labels = fine_batch_labels.cuda(), ekman_batch_labels.cuda(), senti_batch_labels.cuda()
            
            fpred_logits, epred_logits, spred_logits = model(batch_input_ids, batch_attention_mask)
            fpred_distribution = softmax(fpred_logits, 1)
            epred_distribution = softmax(epred_logits, 1)
            spred_distribution = softmax(spred_logits, 1)
            
            fpred_logits_sort = fpred_distribution.sort(descending=True)
            fine_pred_indices = fpred_logits_sort.indices.tolist()[0]

            epred_logits_sort = epred_distribution.sort(descending=True)
            ekman_pred_indices = epred_logits_sort.indices.tolist()[0]
            
            spred_logits_sort = spred_distribution.sort(descending=True)
            senti_pred_indices = spred_logits_sort.indices.tolist()[0]
            
            fp1_list, fp2_list, fp3_list, fr1_list, fr2_list, fr3_list, fpred_list, flabel_list = \
            cal_p_r(fine_pred_indices, fp1_list, fp2_list, fp3_list, fr1_list, fr2_list, fr3_list, fine_batch_labels, fpred_list, flabel_list)

            ep1_list, ep2_list, ep3_list, er1_list, er2_list, er3_list, epred_list, elabel_list = \
            cal_p_r(ekman_pred_indices, ep1_list, ep2_list, ep3_list, er1_list, er2_list, er3_list, ekman_batch_labels, epred_list, elabel_list)

            sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, spred_list, slabel_list  = \
            cal_p_r(senti_pred_indices, sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, senti_batch_labels, spred_list, slabel_list)   
            
        fp1score, fp2score, fp3score = average(fp1_list), average(fp2_list), average(fp3_list)
        ep1score, ep2score, ep3score = average(ep1_list), average(ep2_list), average(ep3_list)
        sp1score, sp2score, sp3score = average(sp1_list), average(sp2_list), average(sp3_list)

        fr1score, fr2score, fr3score = average(fr1_list), average(fr2_list), average(fr3_list)
        er1score, er2score, er3score = average(er1_list), average(er2_list), average(er3_list)
        sr1score, sr2score, sr3score = average(sr1_list), average(sr2_list), average(sr3_list)
        
        result = {}
        result['fine_grained'] = {}
        result['fine_grained']['p'] = [fp1score, fp2score, fp3score]
        result['fine_grained']['r'] = [fr1score, fr2score, fr3score]
        result['fine_grained']['pred_list'] = fpred_list
        result['fine_grained']['label_list'] = flabel_list
        
        result['ekman'] = {}
        result['ekman']['p'] = [ep1score, ep2score, ep3score]
        result['ekman']['r'] = [er1score, er2score, er3score]
        result['ekman']['pred_list'] = epred_list
        result['ekman']['label_list'] = elabel_list
        
        result['sentiment'] = {}
        result['sentiment']['p'] = [sp1score, sp2score, sp3score]
        result['sentiment']['r'] = [sr1score, sr2score, sr3score]        
        result['sentiment']['pred_list'] = spred_list
        result['sentiment']['label_list'] = slabel_list
    return result

def average(score_list):
    score = round(sum(score_list)/len(score_list)*100, 2)
    return score

def cal_p_r(pred_indices, p1_list, p2_list, p3_list, r1_list, r2_list, r3_list, batch_labels, pred_list, label_list):
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
    
    return p1_list, p2_list, p3_list, r1_list, r2_list, r3_list, pred_list, label_list
        
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
    parser.add_argument( "--class_type", type=str, help = "f2c", default = "f2c")
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    