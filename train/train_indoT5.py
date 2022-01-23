import pandas as pd
import os
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration 
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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__=='__main__':
    parser = add_args(argparse.ArgumentParser())
    args = parser.parse_args()

    ## init params
    set_seed(42)
    model_type = "indo-t5"
    batch_size = args.batch_size
    lr = args.lr
    eps = args.eps
    n_epochs = args.n_epochs
    num_beams = args.num_beams
    max_seq_len_amr = args.max_seq_len_amr
    max_seq_len_sent = args.max_seq_len_sent
    result_folder = args.result_folder
    DATA_FOLDER = args.data_folder


    if torch.cuda.is_available():
        device = torch.device("cuda:0") 
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")


    tokenizer = T5Tokenizer.from_pretrained("Wikidepia/IndoT5-base")
    model = T5ForConditionalGeneration.from_pretrained("Wikidepia/IndoT5-base", return_dict=True)
    print(tokenizer)
    print(model.config)
    
    #moving the model to device(GPU/CPU)
    model.to(device)

    # add new vocab (amr special tokens)
    new_tokens_vocab = {}
    new_tokens_vocab['additional_special_tokens'] = []
    for idx, t in enumerate(AMR_TOKENS):
        new_tokens_vocab['additional_special_tokens'].append(t)

    num_added_toks = tokenizer.add_special_tokens(new_tokens_vocab)
    print(f'added {num_added_toks} tokens')

    model.resize_token_embeddings(len(tokenizer))

    # load data

    train_amr_path = os.path.join(DATA_FOLDER, 'train.amr.txt')
    train_sent_path = os.path.join(DATA_FOLDER, 'train.sent.txt')

    dev_amr_path = os.path.join(DATA_FOLDER, 'dev.amr.txt')
    dev_sent_path = os.path.join(DATA_FOLDER, 'dev.sent.txt')

    test_amr_path = os.path.join(DATA_FOLDER, 'test.amr.txt')
    test_sent_path = os.path.join(DATA_FOLDER, 'test.sent.txt')

    train_dataset = AMRToTextDataset(train_amr_path, train_sent_path, tokenizer, 'train')
    dev_dataset = AMRToTextDataset(dev_amr_path, dev_sent_path, tokenizer, 'dev')
    test_dataset = AMRToTextDataset(test_amr_path, test_sent_path, tokenizer, 'test')

    train_loader = AMRToTextDataLoader(dataset=train_dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                                        batch_size=batch_size, shuffle=True)  
    test_loader = AMRToTextDataLoader(dataset=test_dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                                        batch_size=batch_size, shuffle=False)  
    dev_loader = AMRToTextDataLoader(dataset=dev_dataset, model_type=model_type, tokenizer=tokenizer,  max_seq_len_amr=max_seq_len_amr, max_seq_len_sent=max_seq_len_sent, 
                                        batch_size=batch_size, shuffle=False)  

    print('len train dataset: ', str(len(train_dataset)))
    print('len dev dataset: ', str(len(dev_dataset)))
    print('len test dataset:', str(len(test_dataset)))

    print('len train dataloader: ', str(len(train_loader)))
    
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

        list_loss_train.append(total_train_loss/len(train_loader))

        # eval per epoch
        model.eval()
        torch.set_grad_enabled(False)
        list_hyp, list_label = [], []
        
        total_dev_loss = 0

        pbar = tqdm(iter(dev_loader), leave=True, total=len(dev_loader))
        for i, batch_data in enumerate(pbar):
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
            pbar.set_description("(Epoch {}) DEV LOSS:{:.4f} LR:{:.8f}".format((epoch+1),
                    total_dev_loss/(i+1), get_lr(optimizer)))
            
        bleu = calc_corpus_bleu_score(list_hyp, list_label)
        print('bleu score on dev: ', str(bleu))

        list_loss_dev.append(total_dev_loss/len(dev_loader))

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
        if (i<5):
            print('sample: ', list_hyp[i], '----', test_dataset.data['sent'][i])
        list_label.append(test_dataset.data['sent'][i])
    
    ## BLEU SCORE
    bleu = calc_corpus_bleu_score(list_hyp, list_label)
    print('bleu score on test dataset: ', str(bleu))
    with open(os.path.join(result_folder, 'bleu_score_test.txt'), 'w') as f:
        f.write(str(bleu))

    ## save loss data
    with open(os.path.join(result_folder, 'loss_data.tsv'), 'w') as f:
        f.write('train_loss\tval_loss\n')
        for i in range(n_epochs):
            f.write(f'{str(list_loss_train[i])}\t{str(list_loss_dev[i])}\n')

    ## save model
    # torch.save(model.state_dict(), os.path.join(result_folder, "indot5.th"))
    tokenizer.save_pretrained(os.path.join(result_folder, "tokenizer"))
    model.save_pretrained(os.path.join(result_folder, "model"))

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

    
    print(generate("( ketik :ARG0 ( saya ) :ARG1 ( makalah ) ) )", model, tokenizer, num_beams, model_type, 'cpu'))    

    