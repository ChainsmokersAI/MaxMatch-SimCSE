import random

import torch
from torch.utils.data import Dataset

from datasets import load_dataset

from maxmatch_tokenizer import MaxMatchTokenizer

class TrainDataset(Dataset):
    """
    PyTorch Dataset for Vanilla "SimCSE" Training
    """
    def __init__(self, sentences, tokenizer, max_seq_len):
        self.sent=[]
        self.pos=[]

        for sent in sentences:
            # Encode
            enc=tokenizer.encode(sent)
            # Truncate
            if len(enc)>max_seq_len:
                enc=enc[:max_seq_len-1]+[enc[-1]]
            # Append
            self.sent.append(enc)
            self.pos.append(enc)

    def __getitem__(self, idx):
        return self.sent[idx], self.pos[idx]

    def __len__(self):
        return len(self.sent)

class MaxMatchTrainDataset(Dataset):
    """
    PyTorch Dataset for "SimCSE + MaxMatch-Dropout" Training
    """
    def __init__(self, sentences, tokenizer, max_seq_len, p_maxmatch):
        self.sentences=sentences

        # MaxMatch Tokenizer
        _tokenizer=MaxMatchTokenizer()
        _tokenizer.loadBertTokenizer(tokenizer, doNaivePreproc=True)
        self.tokenizer=_tokenizer

        self.max_seq_len=max_seq_len
        self.p_maxmatch=p_maxmatch

    def __getitem__(self, idx):
        # Encode
        enc=self.tokenizer.encode(self.sentences[idx], p=0.0)
        enc_pos=self.tokenizer.encode(self.sentences[idx], p=self.p_maxmatch)

        # Truncate
        if len(enc)>self.max_seq_len:
            enc=enc[:self.max_seq_len-1]+[enc[-1]]
        if len(enc_pos)>self.max_seq_len:
            enc_pos=enc_pos[:self.max_seq_len-1]+[enc_pos[-1]]

        return enc, enc_pos

    def __len__(self):
        return len(self.sentences)

def get_general_sentences():
    """
    Get All Sentences from "Wiki" Data
    """
    sentences=open("./dataset/wiki1m_for_simcse.txt", "r").read().split("\n")
    sentences.remove("")
    
    return sentences

def get_domain_sentences():
    """
    Get All Sentences from "CaseHOLD" Train Set
    """
    contexts=[]
    endings=[]
    for data in load_dataset("lex_glue", "case_hold")["train"]:
        contexts.append(data["context"])
        endings.extend(data["endings"])

    return contexts+endings

def prepare_dataset_for_train(corpus, use_maxmatch, tokenizer, max_seq_len, p_maxmatch):
    """
    Return PyTorch Dataset for Training
    """
    # Load Corpus
    if corpus=="general":
        sentences=get_general_sentences()
    elif corpus=="domain":
        sentences=get_domain_sentences()

    # Load Dataset
    if use_maxmatch=="False":
        dataset=TrainDataset(sentences=sentences, tokenizer=tokenizer, max_seq_len=max_seq_len)
    elif use_maxmatch=="True":
        dataset=MaxMatchTrainDataset(
            sentences=sentences,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            p_maxmatch=p_maxmatch
        )

    return dataset

def get_collate_fn(pad_token_id):
    def collate_fn(batch):
        """
        Same Sequence Length on Same Batch
        """
        max_len_sent=0
        max_len_pos=0
        for sent, pos in batch:
            if len(sent)>max_len_sent: max_len_sent=len(sent)
            if len(pos)>max_len_pos: max_len_pos=len(pos)
                
        batch_sent=[]
        batch_pos=[]
        for sent, pos in batch:
            sent.extend([pad_token_id]*(max_len_sent-len(sent)))
            batch_sent.append(sent)
            
            pos.extend([pad_token_id]*(max_len_pos-len(pos)))
            batch_pos.append(pos)
            
        return torch.tensor(batch_sent), torch.tensor(batch_pos)

    return collate_fn
