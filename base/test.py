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

    
## finetune RoBETa-large
def main():
    """Dataset Loading"""
    class_type = args.class_type # 'fine_grained'
    model_type = 'bert-base-cased'
    
    train_path = './dataset/train.txt'
    dev_path = './dataset/dev.txt'
    test_path = './dataset/test.txt'
    
    test_dataset = goemotion_loader(test_path, model_type)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)
    
    """mapping"""
    fineList = test_dataset.fineList

    ekman_reverse_data = test_dataset.ekman_reverse_data
    ekmanList = test_dataset.ekmanList
    
    senti_reverse_data = test_dataset.senti_reverse_data
    sentiList = test_dataset.sentiList
    
    mappings = [fineList, ekman_reverse_data, ekmanList, senti_reverse_data, sentiList]
    
    """logging and path"""
    save_path = f"./{class_type}"
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    log_path = os.path.join(save_path, 'test.log')
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    """Model Loading"""
    model = EmoModel(model_type, class_type)
    model = model.cuda()    
    modelfile = os.path.join(save_path, 'model.bin')
    model.load_state_dict(torch.load(modelfile))    
    model.eval() 
    
    result = _CalACC(model, test_dataloader, mappings, args)
    fine_grained_result = result['fine_grained']
    ekman_grained_result = result['ekman']
    sentiment_grained_result = result['sentiment']
    
    if args.class_type == 'fine_grained':
        test_pre, test_rec, test_f_macro, test_f_micro = finalscore(fine_grained_result)
        logger.info(f"Fine-grained ## precision: {test_pre}, recall: {test_rec}, macro-F1: {round(test_f_macro*100,2)}, micro-F1: {round(test_f_micro*100,2)}")
        
        test_pre, test_rec, test_f_macro, test_f_micro = finalscore(ekman_grained_result)
        logger.info(f"Ekman ## precision: {test_pre}, recall: {test_rec}, macro-F1: {round(test_f_macro*100,2)}, micro-F1: {round(test_f_micro*100,2)}")
        
        test_pre, test_rec, test_f_macro, test_f_micro = finalscore(sentiment_grained_result)
        logger.info(f"Sentiment ## precision: {test_pre}, recall: {test_rec}, macro-F1: {round(test_f_macro*100,2)}, micro-F1: {round(test_f_micro*100,2)}")
    elif args.class_type == 'ekman':
        test_pre, test_rec, test_f_macro, test_f_micro = finalscore(ekman_grained_result)
        logger.info(f"Ekman ## precision: {test_pre}, recall: {test_rec}, macro-F1: {round(test_f_macro*100,2)}, micro-F1: {round(test_f_micro*100,2)}")
        
        test_pre, test_rec, test_f_macro, test_f_micro = finalscore(sentiment_grained_result)
        logger.info(f"Sentiment ## precision: {test_pre}, recall: {test_rec}, macro-F1: {round(test_f_macro*100,2)}, micro-F1: {round(test_f_micro*100,2)}")
    elif args.class_type == 'sentiment':
        test_pre, test_rec, test_f_macro, test_f_micro = finalscore(sentiment_grained_result)
        logger.info(f"Sentiment ## precision: {test_pre}, recall: {test_rec}, macro-F1: {round(test_f_macro*100,2)}, micro-F1: {round(test_f_micro*100,2)}")        
        
def finalscore(result):
    test_pre = result['p']
    test_rec = result['r']
    test_pred_list = result['pred_list']
    test_label_list = result['label_list']

    _, _, test_f_macro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='macro')
    _, _, test_f_micro, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='micro')   
    
    return test_pre, test_rec, test_f_macro, test_f_micro

def fine2other(fine_ind, mappings):
    fineList, ekman_reverse_data, ekmanList, senti_reverse_data, sentiList = mappings
    emotion = fineList[int(fine_ind)]

    ekman_emotion = ekman_reverse_data[emotion]
    ekman_ind = ekmanList.index(ekman_emotion)

    senti_emotion = senti_reverse_data[emotion]
    senti_ind = sentiList.index(senti_emotion)
    return fine_ind, ekman_ind, senti_ind

def ekman2other(ekman_ind, mappings):
    fineList, ekman_reverse_data, ekmanList, senti_reverse_data, sentiList = mappings
    emotion = ekmanList[int(ekman_ind)]
    
    senti_emotion = senti_reverse_data[emotion]
    senti_ind = sentiList.index(senti_emotion)
    return 0, ekman_ind, senti_ind    

def senti2other(senti_ind, mappings):
    return 0, 0, senti_ind    
    
def _CalACC(model, dataloader, mappings, args):
    model.eval()
    fineList, ekman_reverse_data, ekmanList, senti_reverse_data, sentiList = mappings
    
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
        for i_batch, data in enumerate(tqdm(dataloader)):
            """Prediction"""
            batch_input_ids, batch_attention_mask, fine_batch_labels, ekman_batch_labels, senti_batch_labels = data
            if args.class_type == 'fine_grained':
                batch_labels = [fine_batch_labels, ekman_batch_labels, senti_batch_labels]
                mapping_function = fine2other
            elif args.class_type == 'ekman':
                batch_labels = [ekman_batch_labels, senti_batch_labels]
                mapping_function = ekman2other
            elif args.class_type == 'sentiment':
                batch_labels = [senti_batch_labels]
                mapping_function = senti2other
                
            batch_input_ids, batch_attention_mask = batch_input_ids.cuda(), batch_attention_mask.cuda()
            
            """pred mapping"""
            pred_logits = model(batch_input_ids, batch_attention_mask)
            pred_distribution = softmax(pred_logits, 1) # (1, clsNum)            
            
            fpred_distribution, epred_distribution, spred_distribution =\
                [0 for _ in range(len(fineList))], [0 for _ in range(len(ekmanList))], [0 for _ in range(len(sentiList))]
            
            for ind in range(pred_distribution.shape[1]):                
                prob = pred_distribution[:,ind].item()
                
                fine_ind, ekman_ind, senti_ind = mapping_function(ind, mappings)
                fpred_distribution[fine_ind] += prob
                epred_distribution[ekman_ind] += prob
                spred_distribution[senti_ind] += prob
            fpred_distribution = torch.tensor(fpred_distribution).unsqueeze(0) 
            epred_distribution = torch.tensor(epred_distribution).unsqueeze(0) 
            spred_distribution = torch.tensor(spred_distribution).unsqueeze(0) 
            
            fpred_sort = fpred_distribution.sort(descending=True)
            fine_pred_indices = fpred_sort.indices.tolist()[0]
            
            epred_sort = epred_distribution.sort(descending=True)
            ekman_pred_indices = epred_sort.indices.tolist()[0]
            
            spred_sort = spred_distribution.sort(descending=True)
            senti_pred_indices = spred_sort.indices.tolist()[0]
            
            if args.class_type == 'fine_grained':
                fp1_list, fp2_list, fp3_list, fr1_list, fr2_list, fr3_list, fpred_list, flabel_list = \
                cal_p_r(fine_pred_indices, fp1_list, fp2_list, fp3_list, fr1_list, fr2_list, fr3_list, fine_batch_labels, fpred_list, flabel_list)
                
                ep1_list, ep2_list, ep3_list, er1_list, er2_list, er3_list, epred_list, elabel_list = \
                cal_p_r(ekman_pred_indices, ep1_list, ep2_list, ep3_list, er1_list, er2_list, er3_list, ekman_batch_labels, epred_list, elabel_list)
                
                sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, spred_list, slabel_list  = \
                cal_p_r(senti_pred_indices, sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, senti_batch_labels, spred_list, slabel_list)
            elif args.class_type == 'ekman':
                ep1_list, ep2_list, ep3_list, er1_list, er2_list, er3_list, epred_list, elabel_list = \
                cal_p_r(ekman_pred_indices, ep1_list, ep2_list, ep3_list, er1_list, er2_list, er3_list, ekman_batch_labels, epred_list, elabel_list)
                
                sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, spred_list, slabel_list  = \
                cal_p_r(senti_pred_indices, sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, senti_batch_labels, spred_list, slabel_list)
            elif args.class_type == 'sentiment':
                sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, spred_list, slabel_list  = \
                cal_p_r(senti_pred_indices, sp1_list, sp2_list, sp3_list, sr1_list, sr2_list, sr3_list, senti_batch_labels, spred_list, slabel_list)
            
        
        if args.class_type == 'fine_grained':
            fp1score, fp2score, fp3score = average(fp1_list), average(fp2_list), average(fp3_list)
            ep1score, ep2score, ep3score = average(ep1_list), average(ep2_list), average(ep3_list)
            sp1score, sp2score, sp3score = average(sp1_list), average(sp2_list), average(sp3_list)

            fr1score, fr2score, fr3score = average(fr1_list), average(fr2_list), average(fr3_list)
            er1score, er2score, er3score = average(er1_list), average(er2_list), average(er3_list)
            sr1score, sr2score, sr3score = average(sr1_list), average(sr2_list), average(sr3_list)
        elif args.class_type == 'ekman':
            fp1score, fp2score, fp3score = None, None, None
            ep1score, ep2score, ep3score = average(ep1_list), average(ep2_list), average(ep3_list)
            sp1score, sp2score, sp3score = average(sp1_list), average(sp2_list), average(sp3_list)

            fr1score, fr2score, fr3score = None, None, None
            er1score, er2score, er3score = average(er1_list), average(er2_list), average(er3_list)
            sr1score, sr2score, sr3score = average(sr1_list), average(sr2_list), average(sr3_list)   
        elif args.class_type == 'sentiment':
            fp1score, fp2score, fp3score = None, None, None
            ep1score, ep2score, ep3score = None, None, None
            sp1score, sp2score, sp3score = average(sp1_list), average(sp2_list), average(sp3_list)

            fr1score, fr2score, fr3score = None, None, None
            er1score, er2score, er3score = None, None, None
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
   

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Emotion Classifier" )
    parser.add_argument( "--class_type", type=str, help = "fine_grained or ekman or sentiment", default = "fine_grained")
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    