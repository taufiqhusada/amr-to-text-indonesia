import pandas as pd
import os
import sys
import torch
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
from transformers.optimization import  AdamW, Adafactor 
import time
import warnings
from tqdm import tqdm
from sacrebleu import corpus_bleu
import random
import numpy as np
import argparse

from sklearn.model_selection import KFold
import gc

sys.path.append('../..')
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__=='__main__':
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    ## init params
    set_seed(42)
    model_type = "indo-t5"
    print(args)
    batch_size = args.batch_size
    lr = args.lr
    eps = args.eps
    n_epochs = args.n_epochs
    num_beams = args.num_beams
    max_seq_len_amr = args.max_seq_len_amr
    max_seq_len_sent = args.max_seq_len_sent
    result_folder = args.result_folder
    DATA_FOLDER = args.data_folder
    if (args.resume_from_checkpoint):
        saved_model_folder_path = args.saved_model_folder_path

    
    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")


    # load data
    amr_path = os.path.join(DATA_FOLDER, 'train.amr.txt')
    sent_path = os.path.join(DATA_FOLDER, 'train.sent.txt')

    tokenizer = T5TokenizerFast.from_pretrained("Wikidepia/IndoT5-base")
     # add new vocab (amr special tokens)
    new_tokens_vocab = {}
    new_tokens_vocab['additional_special_tokens'] = tokenizer.additional_special_tokens
    for idx, t in enumerate(AMR_TOKENS):
        new_tokens_vocab['additional_special_tokens'].append(t)

    dataset = AMRToTextDataset(amr_path, sent_path, tokenizer, 'train')
    total_bleu = 0

    for fold,(train_idx,test_idx) in enumerate(KFold.split(dataset, n_splits=5, shuffle=True, random_state=42)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = AMRToTextDataLoader(dataset=train_subsampler, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                        batch_size=batch_size, shuffle=False)
        test_loader = AMRToTextDataLoader(dataset=train_subsampler, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                        batch_size=batch_size, shuffle=False) 

        print('len train dataset: ', str(len(train_subsampler)))
        print('len test dataset: ', str(len(test_subsampler)))

        model = AutoModelForSeq2SeqLM.from_pretrained("Wikidepia/IndoT5-base", return_dict=True)

        #moving the model to device(GPU/CPU)
        model.to(device)

        num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)

        model.resize_token_embeddings(len(tokenizer))
        
        # define optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=lr,
            eps=eps
        )


        # train
        list_loss_train = []
        list_loss_dev = []
        for epoch in range(n_epochs):
            model.train()
            torch.set_grad_enabled(True)
        
            total_train_loss = 0
            list_hyp, list_label = [], []

            train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            for i, batch_data in enumerate(train_pbar):
                enc_batch = torch.LongTensor(batch_data[0]).cuda()
                dec_batch = torch.LongTensor(batch_data[1]).cuda()
                enc_mask_batch = torch.FloatTensor(batch_data[2]).cuda()
                dec_mask_batch = None
                label_batch = torch.LongTensor(batch_data[4]).cuda()
                token_type_batch = None

                outputs = model(input_ids=enc_batch, attention_mask=enc_mask_batch, decoder_input_ids=dec_batch, 
                            decoder_attention_mask=dec_mask_batch, labels=label_batch)
                loss, logits = outputs[:2]
                hyps = logits.topk(1, dim=-1)[1]
                
                loss.backward()
                
                tr_loss = loss.item()
                total_train_loss = total_train_loss + tr_loss
                
                train_pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                        total_train_loss/(i+1), get_lr(optimizer)))
                
                optimizer.step()
                optimizer.zero_grad()


        ## TEST
        model.eval()
        torch.set_grad_enabled(False)

        list_hyp, list_label = [], []

        pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
        for i, batch_data in enumerate(pbar):
            batch_seq = batch_data[-1]

            enc_batch = torch.LongTensor(batch_data[0])
            dec_batch = torch.LongTensor(batch_data[1])
            enc_mask_batch = torch.FloatTensor(batch_data[2])
            dec_mask_batch = None
            label_batch = torch.LongTensor(batch_data[4])
            token_type_batch = None

            # cuda
            enc_batch = enc_batch.cuda()
            dec_batch = dec_batch.cuda()
            enc_mask_batch = enc_mask_batch.cuda() 
            dec_mask_batch = None
            label_batch = label_batch.cuda()
            token_type_batch = None

            hyps = model.generate(input_ids=enc_batch, attention_mask=enc_mask_batch, num_beams=num_beams, max_length=max_seq_len_sent, 
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
            if (i<1):
                print('sample: ', list_hyp[i], '----', test_dataset.data['sent'][i])
            list_label.append(test_dataset.data['sent'][i])
        
        ## BLEU SCORE
        bleu = calc_corpus_bleu_score(list_hyp, list_label)
        print(fold, bleu)
        total_bleu += bleu


        # del model
        del model
        gc.collect()

    print('bleu score avg on all folds: ', str(total_bleu/5))
    with open(os.path.join(result_folder, 'bleu_score_test.txt'), 'w') as f:
        f.write(str(total_bleu/5))