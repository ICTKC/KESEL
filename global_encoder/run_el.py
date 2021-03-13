# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # 指定第1块gpu
import logging
import argparse
import random
from tqdm import tqdm, trange
import simplejson as json

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from knowledge_bert.typing import BertTokenizer as BertTokenizer_label
from knowledge_bert.tokenization import BertTokenizer
from knowledge_bert.modeling import BertForEntityTyping, BertForEntityLinking
from knowledge_bert.optimization import BertAdam
from knowledge_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from eldata_util import DataUtil

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {} n_gpu: {}".format(device, n_gpu))


def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            #if x1[i] > 0 or x1[i] == top:
            if x1[i] > 0:
                yy1.append(i)
            if x2[i] > 0:
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0
        

def train(data_obj, dname, args, embed, model):
    
    data_path = args.data_dir + dname + '_mention_rank'
    local_rep_path = args.local_rep_dir + dname + '_local_rep_mention_rank.npy'
    local_fea_path = args.local_rep_dir + dname + '_local_fea_mention_rank.npy'
    group_path = args.group_path

    mentions, entities, local_feas, ment_names, ment_sents, ment_offsets, ent_ids, mtypes, etypes, pems, labels = \
        data_obj.process_global_data(dname, data_path, local_rep_path, group_path, local_fea_path, args.seq_len, args.candidate_entity_num)

    mention_seq_np, entity_seq_np, local_fea_np, entid_seq_np, pem_seq_np, mtype_seq_np, etype_seq_np, label_seq_np = \
        data_obj.get_local_feature_input(mentions, entities, local_feas, ent_ids, mtypes, etypes, pems, labels, args.seq_len, args.candidate_entity_num)

    seq_tokens_np, seq_tokens_mask_np, seq_tokens_segment_np, seq_ents_np, seq_ents_mask_np, seq_ents_index_np, seq_label_np = \
        data_obj.get_global_feature_input(ment_names, ment_sents, ment_offsets, ent_ids, labels, args.seq_len, args.candidate_entity_num)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
    param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_steps = int(
            len(seq_tokens_np) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                            lr=args.learning_rate,
                            warmup=args.warmup_proportion,
                            t_total=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(seq_tokens_np))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)


    all_seq_input_id = torch.tensor(seq_tokens_np, dtype=torch.long)  # (num_example, 256)
    all_seq_input_mask = torch.tensor(seq_tokens_mask_np, dtype=torch.long) # (num_example, 256)
    all_seq_segment_id = torch.tensor(seq_tokens_segment_np, dtype=torch.long) # (num_example, 256)
    all_seq_input_ent = torch.tensor(seq_ents_np, dtype=torch.long)     # (num_example, 256)
    all_seq_ent_mask = torch.tensor(seq_ents_mask_np, dtype=torch.long)  # (num_example, 256)

    all_seq_label = torch.tensor(seq_label_np, dtype=torch.long)     # (num_example, 3) # 用于hingeloss
    # all_seq_label = torch.tensor(label_seq_np, dtype=torch.long)     # (num_example, 3, 6) #用于BCEloss

    all_seq_mention_rep = torch.tensor(mention_seq_np, dtype=torch.float)  # (num_example, 3, 768)
    all_seq_entity_rep = torch.tensor(entity_seq_np, dtype=torch.float)  # (num_example, 3, 6, 768)
    all_seq_entid = torch.tensor(entid_seq_np, dtype=torch.long) #(num_example, 3, 6)  候选实体的eid
    all_seq_ent_index = torch.tensor(seq_ents_index_np, dtype=torch.long)  # (num_example, 3) eg:[[1,81,141],[],]

    all_seq_pem = torch.tensor(pem_seq_np, dtype=torch.float)  # (num_example, 3, 6)
    all_seq_mtype = torch.tensor(mtype_seq_np, dtype=torch.float) #(num_example, 3, 6, 4) 
    all_seq_etype = torch.tensor(etype_seq_np, dtype=torch.float)  # (num_example, 3, 6, 4) 
    all_seq_local_fea = torch.tensor(local_fea_np, dtype=torch.float)

    train_data = TensorDataset(all_seq_input_id, all_seq_input_mask, all_seq_segment_id, all_seq_input_ent, \
        all_seq_ent_mask, all_seq_ent_index, all_seq_label, \
        all_seq_mention_rep, all_seq_entity_rep, all_seq_entid, all_seq_pem, all_seq_mtype, all_seq_etype, all_seq_local_fea)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    output_loss_file = os.path.join(args.output_dir, "loss")
    loss_fout = open(output_loss_file, 'w')

    output_f1_file = os.path.join(args.output_dir, "result_f1")
    f1_fout = open(output_f1_file, 'w')
    model.train()

    global_step = 0
    best_f1 = -1
    not_better_count = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
        for batch in tqdm(train_dataloader, desc="Iteration"):
            batch = tuple(t.to(device) if i!=3 else t for i, t in enumerate(batch))
            seq_input_id, seq_input_mask, seq_segment_id, seq_input_ent, \
                seq_ent_mask, seq_ent_index, seq_label, \
                    seq_mention_rep, seq_entity_rep, seq_entid, \
                        seq_pem, seq_mtype, seq_etype, seq_local_fea = batch
            seq_input_ent_embed = embed(seq_input_ent+1).to(device)

            # 加一层seq循环 
            # 采样一个周期
            current_input_id_batch = seq_input_id       # shape(batch, ctx_len)
            current_input_mask_batch = seq_input_mask   # shape(b, c)
            current_segment_id_batch = seq_segment_id   # shape(b, c)
            current_input_ent_embed_batch = seq_input_ent_embed     # shape(b, c, dim)
            current_input_ent_batch = seq_input_ent     # shape(b, c)
            current_ent_mask_batch = seq_ent_mask       # shape(b, c)

            for mention_index in range(args.seq_len):
                current_label_batch = seq_label[:, mention_index]                  # shape(b,)
                # current_label_batch = seq_label[:, mention_index, :]               # shape(b, 6)
                current_mention_rep_batch = seq_mention_rep[:, mention_index, :]   # shape(b, 768)
                current_entity_rep_batch = seq_entity_rep[:, mention_index, :, :]  # shape(b, 6, 768)

                current_pem_batch = seq_pem[:, mention_index, :]                   # shape(b, 6)
                current_mtype_batch = seq_mtype[:, mention_index, :, :]            # shape(b, 6, 4)
                current_etype_batch = seq_etype[:, mention_index, :, :]            # shape(b, 6, 4)
                current_local_fea_batch = seq_local_fea[:, mention_index, :]

                current_entid_batch = seq_entid[:, mention_index, :]               # shape(b, 6)
                current_ent_index_batch = seq_ent_index[:, mention_index]          # shape(b, )
                current_entid_embed_batch = embed(current_entid_batch.cpu()+1).to(device) # # shape(b, 6, dim)

                # 训练模型
                loss, scores = \
                    model(current_input_id_batch, current_segment_id_batch, current_input_mask_batch,\
                         current_input_ent_embed_batch, current_ent_mask_batch, current_entid_embed_batch,\
                         current_label_batch, current_mention_rep_batch, current_entity_rep_batch, \
                         current_pem_batch, current_mtype_batch, current_etype_batch, current_local_fea_batch)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # 根据模型的score值，选择预测的实体，修改current_input_ent 和 current_ent_mask
                current_batch_size = current_input_id_batch.size(0)
                pred_ids = torch.argmax(scores, dim=1)  # shape(b)    scores shape(b, 6)
                pred_ids = pred_ids.reshape(current_batch_size, 1) # shape(b, 1)

                pred_entid = torch.gather(current_entid_batch, 1, pred_ids) # shape(b, 1)
                pred_entmask = torch.ones_like(pred_entid) # shape(b, 1)

                alter_input_ent_batch = current_input_ent_batch.scatter(1, current_ent_index_batch.reshape(current_batch_size,1).cpu(), \
                    pred_entid.cpu())
                current_input_ent_embed_batch = embed(alter_input_ent_batch+1).to(device)
                current_ent_mask_batch.scatter_(1, current_ent_index_batch.reshape(current_batch_size,1), \
                    pred_entmask)

                loss.backward()
                loss_fout.write("{}\n".format(loss.item()*args.gradient_accumulation_steps))
                
                tr_loss += loss.item()
                nb_tr_examples += current_input_id_batch.size(0)
                nb_tr_steps += 1

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1                   

                if global_step % 100 == 0:
                    print('global_step: ', global_step, 'global_step loss: ', tr_loss / nb_tr_steps) 
                    dev_f1 = 0
                    dname_list = ['aida-A', 'aida-B', 'msnbc', 'aquaint', 'ace2004', 'clueweb', 'wikipedia']
                    
                    for di, dname in enumerate(dname_list):
                        # test model
                        f1 = predict(data_obj, dname, args, embed, model)
                        print(dname, '\033[92m' + 'micro F1: ' + str(f1) + '\033[0m' ) # 显色
                        f1_fout.write("{}, f1: {}, step: {}\n".format(dname, f1,  global_step))
                        
                        if dname == 'aida-A':
                            dev_f1 = f1
                    if best_f1 < dev_f1:
                        not_better_count = 0
                        best_f1 = dev_f1
                        print('save best model ...')
                        output_model_file = os.path.join(args.output_dir, "pytorch_model_nolocal_{}.bin".format(global_step))
                        torch.save(model.state_dict(), output_model_file)
                    else:
                        not_better_count += 1
                if not_better_count >3:  # 早停
                    exit(0)


        # print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset), flush=True)                                
        # model_to_save = model.module if hasattr(model, 'module') else model
        # output_model_file = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(epoch))
        # torch.save(model.state_dict(), output_model_file)


def predict(data_obj, dname, args, embed, model):
    model.eval()

    data_path = args.data_dir + dname + '_mention_rank'
    local_rep_path = args.local_rep_dir + dname + '_local_rep_mention_rank.npy'
    local_fea_path = args.local_rep_dir + dname + '_local_fea_mention_rank.npy'
    group_path = args.group_path
    
    mentions, entities, local_feas, ment_names, ment_sents, ment_offsets, ent_ids, mtypes, etypes, pems, labels = \
        data_obj.process_global_data(dname, data_path, local_rep_path, group_path, local_fea_path, args.seq_len, args.candidate_entity_num)

    mention_seq_np, entity_seq_np, local_fea_np, entid_seq_np, pem_seq_np, mtype_seq_np, etype_seq_np, label_seq_np = \
        data_obj.get_local_feature_input(mentions, entities, local_feas, ent_ids, mtypes, etypes, pems, labels, args.seq_len, args.candidate_entity_num)

    seq_tokens_np, seq_tokens_mask_np, seq_tokens_segment_np, seq_ents_np, seq_ents_mask_np, seq_ents_index_np, seq_label_np = \
        data_obj.get_global_feature_input(ment_names, ment_sents, ment_offsets, ent_ids, labels, args.seq_len, args.candidate_entity_num)


    # logger.info("***** Running evaluation *****")
    # logger.info("  Num examples = %d", len(seq_tokens_np))
    # logger.info("  Batch size = %d", args.train_batch_size)


    all_seq_input_id = torch.tensor(seq_tokens_np, dtype=torch.long)
    all_seq_input_mask = torch.tensor(seq_tokens_mask_np, dtype=torch.long)
    all_seq_segment_id = torch.tensor(seq_tokens_segment_np, dtype=torch.long)
    all_seq_input_ent = torch.tensor(seq_ents_np, dtype=torch.long)
    all_seq_ent_mask = torch.tensor(seq_ents_mask_np, dtype=torch.long)
    all_seq_label = torch.tensor(seq_label_np, dtype=torch.long)

    all_seq_mention_rep = torch.tensor(mention_seq_np, dtype=torch.float)
    all_seq_entity_rep = torch.tensor(entity_seq_np, dtype=torch.float)
    all_seq_entid = torch.tensor(entid_seq_np, dtype=torch.long)
    all_seq_ent_index = torch.tensor(seq_ents_index_np, dtype=torch.long)

    all_seq_pem = torch.tensor(pem_seq_np, dtype=torch.float)
    all_seq_mtype = torch.tensor(mtype_seq_np, dtype=torch.float)
    all_seq_etype = torch.tensor(etype_seq_np, dtype=torch.float)
    all_seq_local_fea = torch.tensor(local_fea_np, dtype=torch.float)

    eval_data = TensorDataset(all_seq_input_id, all_seq_input_mask, all_seq_segment_id, all_seq_input_ent, \
        all_seq_ent_mask, all_seq_ent_index, all_seq_label, \
        all_seq_mention_rep, all_seq_entity_rep, all_seq_entid, all_seq_pem, all_seq_mtype, all_seq_etype, all_seq_local_fea)
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    all_pred = []
    all_true = []
    for batch in eval_dataloader:
        batch = tuple(t.to(device) if i!=3 else t for i, t in enumerate(batch))
        seq_input_id, seq_input_mask, seq_segment_id, seq_input_ent, \
            seq_ent_mask, seq_ent_index, seq_label, \
                seq_mention_rep, seq_entity_rep, seq_entid, \
                        seq_pem, seq_mtype, seq_etype, seq_local_fea = batch
        seq_input_ent_embed = embed(seq_input_ent+1).to(device)

        # 加一层seq循环  采样一个周期
        current_input_id_batch = seq_input_id
        current_input_mask_batch = seq_input_mask
        current_segment_id_batch = seq_segment_id
        current_input_ent_embed_batch = seq_input_ent_embed
        current_input_ent_batch = seq_input_ent
        current_ent_mask_batch = seq_ent_mask

        for mention_index in range(args.seq_len):
            current_label_batch = seq_label[:, mention_index]
            current_mention_rep_batch = seq_mention_rep[:, mention_index, :]
            current_entity_rep_batch = seq_entity_rep[:, mention_index, :, :]

            current_pem_batch = seq_pem[:, mention_index, :]
            current_mtype_batch = seq_mtype[:, mention_index, :, :]
            current_etype_batch = seq_etype[:, mention_index, :, :]
            current_local_fea_batch = seq_local_fea[:, mention_index, :]

            current_entid_batch = seq_entid[:, mention_index, :]
            current_ent_index_batch = seq_ent_index[:, mention_index]
            current_entid_embed_batch = embed(current_entid_batch.cpu()+1).to(device)

            with torch.no_grad():
                loss, scores = \
                    model(current_input_id_batch, current_segment_id_batch, current_input_mask_batch, current_input_ent_embed_batch, \
                        current_ent_mask_batch, current_entid_embed_batch, current_label_batch, current_mention_rep_batch, current_entity_rep_batch, \
                            current_pem_batch, current_mtype_batch, current_etype_batch, current_local_fea_batch)
                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            # 根据模型的score值，选择预测的实体，修改current_input_ent 和 current_ent_mask
            current_batch_size = current_input_id_batch.size(0)
            pred_ids = torch.argmax(scores, dim=1)
            pred_ids = pred_ids.reshape(current_batch_size, 1)

            pred_entid = torch.gather(current_entid_batch, 1, pred_ids)
            pred_entmask = torch.ones_like(pred_entid)
 
            alter_input_ent_batch = current_input_ent_batch.scatter(1, current_ent_index_batch.reshape(current_batch_size,1).cpu(), \
                pred_entid.cpu())
            current_input_ent_embed_batch = embed(alter_input_ent_batch+1).to(device)
            current_ent_mask_batch.scatter_(1, current_ent_index_batch.reshape(current_batch_size,1), \
                pred_entmask)

            # 记录预测结果
            current_pred = pred_ids.cpu().numpy()
            current_true = current_label_batch.cpu().numpy()
            all_pred.extend(current_pred)
            all_true.extend(current_true)

    # f1
    total_fl_score = compute_precision(all_true, all_pred)
    return total_fl_score

def compute_precision(y, preds):
    num_cands = 6
    num = len(y) / num_cands
    correct = 0
    i = 0
    j = i + num_cands
    group_index = 0
    while 1:
        group_index += 1
        if group_index >= num:
            break
        _y_list = y[i:j]
        _preds_list = preds[i:j]
        _y_index = _y_list.index(min(_y_list))
        _preds_index = _preds_list.index(min(_preds_list))
        if _y_index == _preds_index:
            correct += 1
        i = j
        j = i + num_cands
    preci = float(correct) / num
    return preci

def get_args():
    parser = argparse.ArgumentParser()

    ## path parameters
    parser.add_argument("--data_dir",
                        default="/home/liuyu/kesel/local_encoder/process_data/mention_rank_wiki/",
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--local_rep_dir",
                        default="/home/liuyu/kesel/local_encoder/process_data/local_rep_mention_rank/",
                        type=str,
                        required=False,
                        help="The local context representation of mentions and entities dir.")
    parser.add_argument("--group_path",
                        default="/home/liuyu/kesel/local_encoder/process_data/datasets_statistics.json",
                        type=str,
                        required=False,
                        help="The number of mentions candidates statistics dir")         
    parser.add_argument("--ernie_model", 
                        default="/home/liuyu/kesel/global_encoder/ernie_pretrain/ernie_base", 
                        type=str, 
                        required=False,
                        help="Ernie pre-trained model")  
    parser.add_argument("--output_dir",
                        default="/home/liuyu/kesel/global_encoder/output/",
                        type=str,
                        required=False,
                        help="The output directory where the model predictions.")
    parser.add_argument("--model_path", 
                        default="/home/liuyu/kesel/global_encoder/output/pytorch_model_600.bin", 
                        type=str, 
                        required=False,
                        help="The save path of best model")  

    ## Other parameters
    parser.add_argument("--do_train",
                        default=True,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    ## data_util parameters
    parser.add_argument("--seq_len",
                        default=3,
                        type=int,
                        help="Number of mentions in a segment/sequence.")
    parser.add_argument("--candidate_entity_num",
                        default=6,
                        type=int,
                        help="Number of candidate entity per mention.")
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-3,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    return args

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    print('load entity embedding ...')
    vecs = []
    vecs.append([0]*100)
    with open("/home/liuyu/kesel/global_encoder/kg_embed/entity2vec.vec", 'r') as fin:
        for line in fin:
            vec = line.strip().split('\t')
            vec = [float(x) for x in vec]
            vecs.append(vec)

    embed = torch.FloatTensor(vecs)
    embed = torch.nn.Embedding.from_pretrained(embed)
    logger.info("Shape of entity embedding: "+str(embed.weight.size()))
    del vecs
    
    data_obj = DataUtil()
    dname_train = 'aida-train'
    dname_eval_list = ['aida-A', 'aida-B', 'msnbc', 'aquaint', 'ace2004', 'clueweb', 'wikipedia']

    # 训练模型
    if args.do_train:
        print('train model ...')
        # Prepare model
        model, _ = BertForEntityLinking.from_pretrained(args.ernie_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE)  
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # train model
        train(data_obj, dname_train, args, embed, model)

    elif args.do_eval:
        print('predict model ...', args.model_path)
        # Prepare model
        model_state_dict = torch.load(args.model_path)
        model, _ = BertForEntityLinking.from_pretrained(args.ernie_model, state_dict=model_state_dict)
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        # predict model
        for di, dname_eval in enumerate(dname_eval_list):
            f1 = predict(data_obj, dname_eval, args, embed, model)
            print(dname_eval, '\033[92m' + 'micro F1: ' + str(f1) + '\033[0m' ) # 显色

if __name__ == "__main__":
    main()
