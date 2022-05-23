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

sys.path.append('../../..')
from utils.constants import AMR_TOKENS
from utils.data_utils import AMRToTextDataset, AMRToTextDataLoader
from utils.scoring import calc_corpus_bleu_score
from utils.eval import generate
from utils.utils_argparser import add_args

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

    if (args.resume_from_checkpoint):
        print('resume from checkpoint')
        tokenizer = T5TokenizerFast.from_pretrained(os.path.join(saved_model_folder_path, 'tokenizer'))
    else:
        tokenizer = T5TokenizerFast.from_pretrained("Wikidepia/IndoT5-base")

    # add new vocab (amr special tokens)
    new_tokens_vocab = {}
    new_tokens_vocab['additional_special_tokens'] = tokenizer.additional_special_tokens
    for idx, t in enumerate(AMR_TOKENS):
        new_tokens_vocab['additional_special_tokens'].append(t)

    dataset = AMRToTextDataset(amr_path, sent_path, tokenizer, 'train')
    total_bleu = 0

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    list_avg_loss_train_per_epoch = [0 for i in range(n_epochs)]
    list_avg_loss_test_per_epoch = [0 for i in range(n_epochs)]
    list_avg_bleu_per_epoch = [0 for i in range(n_epochs)]

    for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = AMRToTextDataLoader(dataset=dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                        batch_size=batch_size, shuffle=False, sampler=train_subsampler)
        test_loader = AMRToTextDataLoader(dataset=dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                        batch_size=batch_size, shuffle=False, sampler=test_subsampler) 

        print('len train dataset and dataloader: ', str(len(train_subsampler)), str(len(train_loader)))
        print('len test dataset and dataloader: ', str(len(test_subsampler)), str(len(test_loader)))

        if (args.resume_from_checkpoint):
            model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(saved_model_folder_path, 'model'))
        else:
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

            list_avg_loss_train_per_epoch[epoch] += total_train_loss/len(train_loader)

            ## test (calc bleu and loss)
            model.eval()
            torch.set_grad_enabled(False)

            list_hyp, list_label = [], []
            total_dev_loss = 0

            pbar = tqdm(iter(test_loader), leave=True, total=len(test_loader))
            for batch_data in pbar:
                batch_seq = batch_data[-1]

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

                batch_list_hyp = []
                batch_list_label = []
                for j in range(len(hyps)):
                    hyp = hyps[j,:].squeeze()
                    label = label_batch[j,:].squeeze()

                    batch_list_hyp.append(tokenizer.decode(hyp, skip_special_tokens=True))
                    batch_list_label.append(tokenizer.decode(label[label != -100], skip_special_tokens=True))

                list_hyp += batch_list_hyp
                list_label += batch_list_label
                
                total_dev_loss += loss.item()

                pbar.set_description("(Epoch {}) TEST LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                        total_dev_loss/(i+1), get_lr(optimizer)))
            
            list_avg_loss_test_per_epoch[epoch] += total_dev_loss/len(test_loader)

            for i in range(len(list_hyp)):
                if (i<1):
                    print('sample: ', list_hyp[i], '----', list_label[i])
            
            ## BLEU SCORE
            bleu = calc_corpus_bleu_score(list_hyp, list_label)
            list_avg_bleu_per_epoch[epoch] += bleu

        # del model
        del model
        gc.collect()

    for i in range(n_epochs):
        list_avg_loss_train_per_epoch[i] /= 5
        list_avg_loss_test_per_epoch[i] /= 5
        list_avg_bleu_per_epoch[i] /= 5
    
    print('avg loss train: ', list_avg_loss_train_per_epoch)
    print('avg loss test ', list_avg_loss_test_per_epoch)
    print('avg bleu', list_avg_bleu_per_epoch)