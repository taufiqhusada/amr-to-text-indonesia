import pandas as pd
import os
import sys
import torch
from custom_transformers.models.t5 import T5ForConditionalGeneration, T5TokenizerFast
from transformers.optimization import  AdamW, Adafactor 
import time
import warnings
from tqdm import tqdm
from sacrebleu import corpus_bleu
import random
import numpy as np
import argparse

sys.path.append('..')
from utils.constants import AMR_TOKENS
from utils.data_utils import AMRToTextDataset, AMRToTextDataLoader
from utils.scoring import calc_corpus_bleu_score
from utils.eval import generate
from utils.utils_argparser import add_args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__=='__main__':
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()
    set_seed(42)

    model_type = "indo-t5"
    saved_model_folder_path = args.saved_model_folder_path
    data_folder = args.data_folder
    batch_size = args.batch_size
    max_seq_len_amr = args.max_seq_len_amr
    max_seq_len_sent = args.max_seq_len_sent
    result_folder = args.result_folder
    num_beams = args.num_beams

    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    tokenizer = T5TokenizerFast.from_pretrained(os.path.join(saved_model_folder_path, 'tokenizer'))
    model = T5ForConditionalGeneration.from_pretrained(os.path.join(saved_model_folder_path, 'model'), additional_config = {'tree_max':max_seq_len_amr})
    print(tokenizer)
    print(model.config)

    #moving the model to device(GPU/CPU)
    model.to(device)

    test_amr_path = os.path.join(data_folder, 'test.amr.txt')
    test_sent_path = os.path.join(data_folder, 'test.sent.txt')
    test_level_path = os.path.join(data_folder, 'test.amr.txt.tree_level')  
    test_dataset = AMRToTextDataset(test_amr_path, test_sent_path, tokenizer, 'test', test_level_path)
    test_loader = AMRToTextDataLoader(dataset=test_dataset, model_type=model_type, tokenizer=tokenizer, with_tree_level=True,  
                        max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, batch_size=batch_size, shuffle=False)

    print('len test dataset:', str(len(test_dataset)))

    model.eval()
    torch.set_grad_enabled(False)

    list_hyp, list_label = [], []

    pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
    for i, batch_data in enumerate(pbar):
        batch_seq = batch_data[-1]

        enc_batch = torch.LongTensor(batch_data[0]).cuda()
        dec_batch = torch.LongTensor(batch_data[1]).cuda()
        enc_mask_batch = torch.FloatTensor(batch_data[2]).cuda()
        dec_mask_batch = None
        label_batch = torch.LongTensor(batch_data[4]).cuda()
        level_batch = torch.LongTensor(batch_data[5]).cuda()

        hyps = model.generate(input_ids=enc_batch,  tree_ids = level_batch, attention_mask=enc_mask_batch, num_beams=num_beams, max_length=max_seq_len_sent, 
                            early_stopping=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)


        batch_list_hyp = []
        batch_list_label = []
        for j in range(len(hyps)):
            hyp = hyps[j]
            label = label_batch[j,:].squeeze()
        
            batch_list_hyp.append(tokenizer.decode(hyp, skip_special_tokens=True))
            batch_list_label.append(tokenizer.decode(label[label != -100], skip_special_tokens=True))
        
        list_hyp += batch_list_hyp
        list_label += batch_list_label

    list_label = []
    for i in range(len(list_hyp)):
        if (i<10):
            print('sample: ', list_hyp[i], '----', test_dataset.data['sent'][i])
        list_label.append(test_dataset.data['sent'][i])
    
    ## BLEU SCORE
    bleu = calc_corpus_bleu_score(list_hyp, list_label)
    print('bleu score on test dataset: ', str(bleu))
    with open(os.path.join(result_folder, 'bleu_score_test.txt'), 'w') as f:
        f.write(str(bleu))  

    ## save generated outputs
    with open(os.path.join(result_folder, 'test_generations.txt'), 'w') as f:
        for i in range(len(list_hyp)):
            e = list_hyp[i]
            f.write(e)
            if (i != len(list_hyp)-1):
                f.write('\n')
            
    ## save label 
    with open(os.path.join(result_folder, 'test_label.txt'), 'w') as f:
        for i in range(len(list_label)):
            e = list_label[i]
            f.write(e)
            if (i != len(list_label)-1):
                f.write('\n')