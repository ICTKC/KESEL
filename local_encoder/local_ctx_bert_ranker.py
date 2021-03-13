
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel, BertTokenizer
import time
import io
import json
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LocalCtxBertRanker(nn.Module):
    def __init__(self, config):
        super(LocalCtxBertRanker, self).__init__()
        self.hidden_size = 768
        self.hidden_dim = 256
        self.hid_dims = config['hid_dims']
        self.dr = config['dr']
        # Typing feature
        self.type_matrix = torch.nn.Parameter(torch.randn([4, 5]))

        # load pre-train model bert
        self.tokenizer = BertTokenizer.from_pretrained('/home/liuyu/kesel/local_encoder/bert_pretrain/')
        self.bert = BertModel.from_pretrained('/home/liuyu/kesel/local_encoder/bert_pretrain/')      
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.score_bert = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size , self.hidden_dim), 
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dr), 
                torch.nn.Linear(self.hidden_dim, 1))
        
        self.score_combine = torch.nn.Sequential(
                torch.nn.Linear(5, self.hid_dims), 
                torch.nn.ReLU(),
                torch.nn.Dropout(p=self.dr), 
                torch.nn.Linear(self.hid_dims, 1))


    def forward(self, ment_ctx_tokens, ent_desc_tokens, token_ids, entity_ids, mtype, etype, p_e_m, name_dis): 
        n_ments, n_cands = entity_ids.size() 

        encoded_inputs = self.tokenizer(ment_ctx_tokens, ent_desc_tokens, padding=True, truncation=True, max_length=240, return_tensors="pt")
        input_ids = encoded_inputs['input_ids'].to(DEVICE)
        token_type_ids = encoded_inputs['token_type_ids'].to(DEVICE)
        attention_mask = encoded_inputs['attention_mask'].to(DEVICE)

        last_hidden, _ = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)  
        local_ent_scores = self.score_bert(last_hidden[:,0,:]) 

        # Typing feature  
        self.mt_emb = torch.matmul(mtype, self.type_matrix).view(n_ments, 1, -1)  # shape (b,1,5)
        self.et_emb = torch.matmul(etype.view(-1, 4), self.type_matrix).view(n_ments, n_cands, -1) # shape (b,8,5)
        tm = torch.sum(self.mt_emb*self.et_emb, -1, True)  # shape (b,8)
 
        inputs = torch.cat([local_ent_scores.view(n_ments * n_cands, -1), 
                            # torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1),
                            # tm.view(n_ments * n_cands, -1),
                            # name_dis.view(n_ments * n_cands, -1)
                            ] , dim=1)
        inputs = inputs.repeat(1,5)  #shape(b,20)
        scores = self.score_combine(inputs).view(n_ments, n_cands) 

        return scores
