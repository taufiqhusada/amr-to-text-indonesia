import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

T5_PREFIX = "translate graph to indonesian: "

# class to load preprocessed amr data
class AMRToTextDataset(Dataset):    
    def __init__(self, file_amr_path, file_sent_path, tokenizer, split):
        temp_list_amr_input = []
        with open(file_amr_path) as f:
            temp_list_amr_input = f.readlines()
        list_amr_input = []
        for item in temp_list_amr_input:
            list_amr_input.append(item.strip().lower())  # lowercase for bart tokenizer
            
        temp_list_sent_output = []
        with open(file_sent_path) as f:
            temp_list_sent_output = f.readlines()
        list_sent_output = []
        for item in temp_list_sent_output:
            list_sent_output.append(item.strip().lower())
        
        df = pd.DataFrame(list(zip(list_amr_input, list_sent_output)), columns = ['amr','sent'])
        self.data = df
        self.tokenizer = tokenizer
 
    
    def __getitem__(self, index):
        data = self.data.loc[index,:]
        amr, sent = data['amr'], data['sent']
       
        tokenize_amr = self.tokenizer.encode(amr, add_special_tokens=False)
        tokenize_sent = self.tokenizer.encode(sent, add_special_tokens=False)
        
        item = {'input':{}, 'output':{}}
        item['input']['encoded'] = tokenize_amr
        item['input']['raw'] = amr
        item['output']['encoded'] = tokenize_sent
        item['output']['raw'] = sent
        return item
    
    def __len__(self):
        return len(self.data)
        
## Data loader class
## This dataloader class for T5 and BART is using code (but modified a little bit) from https://github.com/indobenchmark/indonlg/blob/master/utils/data_utils.py
class AMRToTextDataLoader(DataLoader):
    def __init__(self, max_seq_len_amr=512, max_seq_len_sent=384, label_pad_token_id=-100, model_type='indo-t5', tokenizer=None, *args, **kwargs):
        super(AMRToTextDataLoader, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.max_seq_len_amr = max_seq_len_amr
        self.max_seq_len_sent = max_seq_len_sent
        
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        
        self.label_pad_token_id = label_pad_token_id
        
        if model_type == 'indo-t5':
            self.bos_token_id = tokenizer.pad_token_id
            self.t5_prefix =np.array(self.tokenizer.encode(T5_PREFIX, add_special_tokens=False))
            self.collate_fn = self._t5_collate_fn
        elif model_type == 'indo-bart':
            source_lang = "[indonesia]"
            target_lang = "[indonesia]"

            self.src_lid_token_id = tokenizer.special_tokens_to_ids[source_lang]
            self.tgt_lid_token_id = tokenizer.special_tokens_to_ids[target_lang]
            self.collate_fn = self._bart_collate_fn
        else:
            raise ValueError(f'Unknown model_type `{model_type}`')
            
    def _t5_collate_fn(self, batch):
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len_amr, max(map(lambda x: len(x['input']['encoded']), batch))  + len(self.t5_prefix))
        max_dec_len = min(self.max_seq_len_sent, max(map(lambda x: len(x['output']['encoded']), batch)) + 1)
        
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, item in enumerate(batch):
            input_seq = item['input']['encoded']
            label_seq = item['output']['encoded']
            input_seq, label_seq = input_seq[:max_enc_len - len(self.t5_prefix)], label_seq[:max_dec_len - 1]
            
            # Assign content
            enc_batch[i,len(self.t5_prefix):len(self.t5_prefix) + len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + len(self.t5_prefix)] = 1
            dec_mask_batch[i,:len(label_seq) + 1] = 1
            
            # Assign special token to encoder input
            enc_batch[i,:len(self.t5_prefix)] = self.t5_prefix
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.bos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            
        
        return enc_batch, dec_batch, enc_mask_batch, None, label_batch
    
    def _bart_collate_fn(self, batch):
        # encoder input
        # <sent><eos><langid>
        # decoder input - 
        # <langid><sent><eos>
        # decoder output
        # <sent><eos><langid>
        
        batch_size = len(batch)
        max_enc_len = min(self.max_seq_len_amr, max(map(lambda x: len(x['input']['encoded']), batch)) + 2) # + 2 for eos and langid
        max_dec_len = min(self.max_seq_len_sent, max(map(lambda x: len(x['output']['encoded']), batch)) + 2) # + 2 for eos and langid
        
        enc_batch = np.full((batch_size, max_enc_len), self.pad_token_id, dtype=np.int64)
        dec_batch = np.full((batch_size, max_dec_len), self.pad_token_id, dtype=np.int64)
        label_batch = np.full((batch_size, max_dec_len), self.label_pad_token_id, dtype=np.int64)
        enc_mask_batch = np.full((batch_size, max_enc_len), 0, dtype=np.float32)
        dec_mask_batch = np.full((batch_size, max_dec_len), 0, dtype=np.float32)
        
        for i, item in enumerate(batch):
            input_seq = item['input']['encoded']
            label_seq = item['output']['encoded']
            input_seq, label_seq = input_seq[:max_enc_len-2], label_seq[:max_dec_len - 2]
            
            # Assign content
            enc_batch[i,0:len(input_seq)] = input_seq
            dec_batch[i,1:1+len(label_seq)] = label_seq
            label_batch[i,:len(label_seq)] = label_seq
            enc_mask_batch[i,:len(input_seq) + 2] = 1
            dec_mask_batch[i,:len(label_seq) + 2] = 1
            
            # Assign special token to encoder input
            enc_batch[i,len(input_seq)] = self.eos_token_id
            enc_batch[i,1+len(input_seq)] = self.src_lid_token_id
            
            # Assign special token to decoder input
            dec_batch[i,0] = self.tgt_lid_token_id
            dec_batch[i,1+len(label_seq)] = self.eos_token_id
            
            # Assign special token to label
            label_batch[i,len(label_seq)] = self.eos_token_id
            label_batch[i,1+len(label_seq)] = self.tgt_lid_token_id
        
        return enc_batch, dec_batch, enc_mask_batch, None, label_batch