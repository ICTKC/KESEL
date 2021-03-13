import utils
import ntee
import dataset as D
from vocabulary import Vocabulary
from abstract_word_entity import load as load_model
from local_ctx_bert_ranker import LocalCtxBertRanker
from local_ctx_att_ranker import LocalCtxAttRanker

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import copy
import csv
import json
import time
import numpy as np
import Levenshtein
from random import shuffle
from pprint import pprint
from itertools import count

ModelClass = LocalCtxAttRanker
# ModelClass = LocalCtxBertRanker

wiki_prefix = 'en.wikipedia.org/wiki/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EDRanker:
    """
    ranking candidates
    """

    def __init__(self, config):
        print('--- create model ---')

        config['entity_embeddings'] = config['entity_embeddings'] / \
                                      np.maximum(np.linalg.norm(config['entity_embeddings'],
                                                                axis=1, keepdims=True), 1e-12)
        config['entity_embeddings'][config['entity_voca'].unk_id] = 1e-10
        config['word_embeddings'] = config['word_embeddings'] / \
                                    np.maximum(np.linalg.norm(config['word_embeddings'],
                                                              axis=1, keepdims=True), 1e-12)
        config['word_embeddings'][config['word_voca'].unk_id] = 1e-10
        self.word_vocab = config['word_voca']
        self.ent_vocab = config['entity_voca']

        print('prerank model')
        self.prerank_model = ntee.NTEE(config)
        self.args = config['args']

        print('main model')
        if self.args.mode == 'eval':
            print('try loading model from', self.args.model_path)
            self.model = ModelClass(config)
            self.model.load_state_dict(torch.load(self.args.model_path + '.state_dict'))
        else:
            print('create new model')
            self.model = ModelClass(config)

        print('load entity desc')
        self.load_ent_desc(500, 3)

        self.prerank_model.to(DEVICE)
        self.model.to(DEVICE)

    def load_ent_desc(self, max_desc_len, n_grams):
        ent_desc = json.load(open('/home/liuyu/kesel/data/ent2desc.json', 'r'))
        print('entdesc length:', len(ent_desc))
        self.ent_desc_token = [[] for i in range(self.ent_vocab.size())]

        for ent in ent_desc:
            min_len = min(len(ent_desc[ent]), max_desc_len)
            self.ent_desc_token[self.ent_vocab.get_id(wiki_prefix + ent)] = ent_desc[ent][0:min_len]

    def get_data_items(self, dataset, predict=False):
        data = []
        cand_source = 'candidates'
        n_ments = 0 
        for doc_name, content in dataset.items():
            items = []
            conll_doc = content[0].get('conll_doc', None)

            for m in content:
                try:
                    named_cands = [c[0] for c in m[cand_source]]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m[cand_source]]
                    etype = [c[2] for c in m[cand_source]]
                except:
                    named_cands = [c[0] for c in m['candidates']]
                    p_e_m = [min(1., max(1e-3, c[1])) for c in m['candidates']]
                    etype = [c[2] for c in m['candidates']]
                try:
                    true_pos = named_cands.index(m['gold'][0])
                    p = p_e_m[true_pos]
                except:
                    true_pos = -1

                named_cands = named_cands[:min(self.args.n_cands_before_rank, len(named_cands))]
                p_e_m = p_e_m[:min(self.args.n_cands_before_rank, len(p_e_m))]
                etype = etype[:min(self.args.n_cands_before_rank, len(etype))]
                if true_pos >= len(named_cands):
                    if not predict:
                        true_pos = len(named_cands) - 1
                        p_e_m[-1] = p
                        named_cands[-1] = m['gold'][0]
                    else:
                        true_pos = -1

                cands = [self.ent_vocab.get_id(wiki_prefix + c) for c in named_cands]
                mask = [1.] * len(cands)
                if len(cands) == 0 and not predict:
                    continue
                elif len(cands) < self.args.n_cands_before_rank:
                    cands += [self.ent_vocab.unk_id] * (self.args.n_cands_before_rank - len(cands))
                    etype += [[0, 0, 0, 1]] * (self.args.n_cands_before_rank - len(etype))
                    named_cands += [Vocabulary.unk_token] * (self.args.n_cands_before_rank - len(named_cands))
                    p_e_m += [1e-8] * (self.args.n_cands_before_rank - len(p_e_m))
                    mask += [0.] * (self.args.n_cands_before_rank - len(mask))

                lctx = m['context'][0].strip().split()
                lctx_ids = [self.prerank_model.word_voca.get_id(t) for t in lctx if utils.is_important_word(t)]
                lctx_ids = [tid for tid in lctx_ids if tid != self.prerank_model.word_voca.unk_id]
                lctx_ids = lctx_ids[max(0, len(lctx_ids) - self.args.ctx_window//2):]

                lctx_tokens = [t for t in lctx if utils.is_important_word(t)]
                lctx_tokens = [t for t in lctx_tokens if t != '#UNK#']
                lctx_tokens = lctx_tokens[max(0, len(lctx_tokens) - self.args.ctx_window//2):]

                rctx = m['context'][1].strip().split()
                rctx_ids = [self.prerank_model.word_voca.get_id(t) for t in rctx if utils.is_important_word(t)]
                rctx_ids = [tid for tid in rctx_ids if tid != self.prerank_model.word_voca.unk_id]
                rctx_ids = rctx_ids[:min(len(rctx_ids), self.args.ctx_window//2)]

                rctx_tokens = [t for t in rctx if utils.is_important_word(t)]
                rctx_tokens = [t for t in rctx_tokens if t != '#UNK#']
                rctx_tokens = rctx_tokens[max(0, len(rctx_tokens) - self.args.ctx_window//2):]
        
                ment = m['mention'].strip().split()
                ment_ids = [self.prerank_model.word_voca.get_id(t) for t in ment if utils.is_important_word(t)]
                ment_ids = [tid for tid in ment_ids if tid != self.prerank_model.word_voca.unk_id]
                ment_name = m['mention'].strip()

                context_tokens = ' '.join(lctx_tokens) + ' '+ m['mention'] + ' ' + ' '.join(rctx_tokens)
                m['sent'] = ' '.join(lctx + rctx)
                mtype = m['mtype']
                # secondary local context (for computing relation scores)
                if conll_doc is not None:
                    conll_m = m['conll_m']
                    sent = conll_doc['sentences'][conll_m['sent_id']]
                    start = conll_m['start']
                    end = conll_m['end']
                
                    ment_offset_start = conll_m['start']
                    ment_offset_end = conll_m['end']
                    if len(sent) > 18:  # 超参数
                        ment_ctx_token = sent
                    else:
                        if conll_m['sent_id'] == 0: 
                            ment_ctx_token = sent + conll_doc['sentences'][conll_m['sent_id'] + 1]
                        else:
                            ment_ctx_token = conll_doc['sentences'][conll_m['sent_id'] - 1] + sent 
                            ment_offset_start += len(conll_doc['sentences'][conll_m['sent_id'] - 1])
                            ment_offset_end += len(conll_doc['sentences'][conll_m['sent_id'] - 1])
                    # ment_ctx_token = [token for token in ment_ctx_token if utils.is_important_word(token)]
               
                else:
                    ment_ctx_token = ['unk']
                    ment_offset_start = 0
                    ment_offset_end = 0

                items.append({'context': (lctx_ids, rctx_ids),
                              'ment_ids': ment_ids,
                              'context_tokens': context_tokens,
                              'ment_ctx_token': ment_ctx_token,
                              'ment_offset_start': ment_offset_start,
                              'ment_offset_end': ment_offset_end,
                              'ment_name':ment_name,
                              'cands': cands,
                              'named_cands': named_cands,
                              'p_e_m': p_e_m,
                              'mask': mask,
                              'true_pos': true_pos,
                              'mtype': mtype,
                              'etype': etype, 
                              'doc_name': doc_name,
                              'raw': m
                              })
                n_ments += 1
            if len(items) > 0:
                if len(items) > 64: # when bert model, set batch to 16
                    for k in range(0, len(items), 64):
                        data.append(items[k:min(len(items), k + 64)])
                else:
                    data.append(items)
        return self.prerank(data, predict)
    
    def prerank(self, dataset, predict=False):
        new_dataset = []
        has_gold = 0
        total = 0

        for content in dataset:
            items = []

            if self.args.keep_ctx_ent > 0:
                # rank the candidates by ntee scores
                lctx_ids = [m['context'][0][max(len(m['context'][0]) - self.args.prerank_ctx_window // 2, 0):]
                            for m in content]
                rctx_ids = [m['context'][1][:min(len(m['context'][1]), self.args.prerank_ctx_window // 2)]
                            for m in content]
                ment_ids = [[] for m in content]
                token_ids = [l + m + r if len(l) + len(r) > 0 else [self.prerank_model.word_voca.unk_id]
                             for l, m, r in zip(lctx_ids, ment_ids, rctx_ids)]

                entity_ids = [m['cands'] for m in content]
                entity_ids = Variable(torch.LongTensor(entity_ids).to(DEVICE))

                entity_mask = [m['mask'] for m in content]
                entity_mask = Variable(torch.FloatTensor(entity_mask).to(DEVICE))

                token_ids, token_offsets = utils.flatten_list_of_lists(token_ids)
                token_offsets = Variable(torch.LongTensor(token_offsets).to(DEVICE))
                token_ids = Variable(torch.LongTensor(token_ids).to(DEVICE))

                log_probs = self.prerank_model.forward(token_ids, token_offsets, entity_ids, use_sum=True)
                log_probs = (log_probs * entity_mask).add_((entity_mask - 1).mul_(1e10))
                _, top_pos = torch.topk(log_probs, dim=1, k=self.args.keep_ctx_ent)
                top_pos = top_pos.data.cpu().numpy()

            else:
                top_pos = [[]] * len(content)

            # select candidats: mix between keep_ctx_ent best candidates (ntee scores) with
            # keep_p_e_m best candidates (p_e_m scores)
            for i, m in enumerate(content):
                sm = {'cands': [],
                      'named_cands': [],
                      'p_e_m': [],
                      'mask': [],
                      'etype': [],
                      'true_pos': -1}
                m['selected_cands'] = sm

                selected = set(top_pos[i])
                idx = 0
                while len(selected) < self.args.keep_ctx_ent + self.args.keep_p_e_m:
                    if idx not in selected:
                        selected.add(idx)
                    idx += 1

                selected = sorted(list(selected))
                for idx in selected:
                    if idx>len(m['cands'])-1:
                        continue
                    sm['cands'].append(m['cands'][idx])
                    sm['named_cands'].append(m['named_cands'][idx])
                    sm['p_e_m'].append(m['p_e_m'][idx])
                    sm['mask'].append(m['mask'][idx])
                    sm['etype'].append(m['etype'][idx])
                    if idx == m['true_pos']:
                        sm['true_pos'] = len(sm['cands']) - 1

                if not predict:
                    if sm['true_pos'] == -1:
                        continue
                items.append(m)
                if sm['true_pos'] >= 0:
                    has_gold += 1
                total += 1

                if predict:
                    # only for oracle model, not used for eval
                    if sm['true_pos'] == -1:
                        sm['true_pos'] = 0  # a fake gold, happens only 2%, but avoid the non-gold

            if len(items) > 0:
                new_dataset.append(items)

        print('recall', has_gold / total)
        return new_dataset

    def train(self, org_train_dataset, org_dev_datasets, config):
        print('extracting training data')
        train_dataset = self.get_data_items(org_train_dataset, predict=False)
        print('#train docs', len(train_dataset))

        dev_datasets = []
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, self.get_data_items(data, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

        print('creating optimizer')
        optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=config['lr'])
        best_f1 = -1
        best_f1_datasets = []
        not_better_count = 0
        is_counting = False
        eval_after_n_epochs = self.args.eval_after_n_epochs

        for e in range(config['n_epochs']):
            shuffle(train_dataset)

            total_loss = 0
            for dc, batch in enumerate(train_dataset):  # each document is a minibatch
                self.model.train()
                optimizer.zero_grad()

                # convert data items to pytorch inputs
                token_ids = [m['context'][0] + m['context'][1]
                             if len(m['context'][0]) + len(m['context'][1]) > 0
                             else [self.word_vocab.unk_id]
                             for m in batch]
                token_ids, token_mask = utils.make_equal_len(token_ids, self.word_vocab.unk_id)
                token_ids = Variable(torch.LongTensor(token_ids).to(DEVICE))
                token_mask = Variable(torch.FloatTensor(token_mask).to(DEVICE))

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).to(DEVICE))
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).to(DEVICE))

                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).to(DEVICE))
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).to(DEVICE))
                mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).to(DEVICE))
                etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).to(DEVICE))  
 
                n_ments = len(batch)
                n_cands = self.args.keep_ctx_ent + self.args.keep_p_e_m

                ment_name = [m['ment_name'] for m in batch]
                cands_name = [m['named_cands'] for m in batch]
                name_dis = [[0 for _ in range(n_cands)] for _ in range(n_ments)]
                for i in range(n_ments):
                    for j in range(n_cands):
                        # string edit distance 
                        name_dis[i][j] = Levenshtein.jaro_winkler(ment_name[i], cands_name[i][j])                        
                name_dis = Variable(torch.FloatTensor(name_dis).to(DEVICE)) 

                context_tokens = [ m['context_tokens'] for m in batch]
                context_tokens = np.array(context_tokens).repeat(entity_ids.shape[1]).tolist()  # shape(batch, 8, len_token)

                ment_ctx_tokens = [ ' '.join(m['ment_ctx_token']) for m in batch]
                ent_desc_tokens = [[[] for _ in range(n_cands)] for _ in range(n_ments)]
                for i in range(n_ments):
                    for j in range(n_cands):
                        ent_desc_tokens[i][j] = ' '.join(self.ent_desc_token[batch[i]['selected_cands']['cands'][j]])
                ment_ctx_tokens = np.array(ment_ctx_tokens).repeat(entity_ids.shape[1]).tolist()
                ent_desc_tokens = np.array(ent_desc_tokens).reshape(-1).tolist()

                # local model : natt + tom
                scores, mention_rep, entity_rep = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, name_dis)

                # local model : bert
                # scores = self.model.forward(context_tokens, ent_desc_tokens, token_ids, entity_ids, mtype, etype, p_e_m, name_dis)  # scores shape(batch, n_cands)

                loss = F.multi_margin_loss(scores, true_pos, margin=0.1)  # scores shape(b,n_cands), true_pos shape(b)

                loss.backward()
                optimizer.step()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                loss = loss.cpu().data.numpy()
                total_loss += loss                    
            print('epoch', e, 'total loss', total_loss, total_loss / len(train_dataset))

            if (e + 1) % eval_after_n_epochs == 0:
                dev_f1 = 0
                temp_rlt = []
                for di, (dname, data) in enumerate(dev_datasets):
                    predictions = self.predict(data)
                    f1 = D.eval(org_dev_datasets[di][1], predictions)
                    print(dname, utils.tokgreen('micro F1: ' + str(f1)))
                    temp_rlt.append([dname, f1])
                    if dname == 'aida-A':
                        dev_f1 = f1

                if config['lr'] == 1e-4 and dev_f1 >= self.args.dev_f1_change_lr:
                    eval_after_n_epochs = 2
                    is_counting = True
                    best_f1 = dev_f1
                    not_better_count = 0

                    config['lr'] = 1e-5
                    print('change learning rate to', config['lr'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = config['lr']

                if dev_f1 < best_f1:
                    not_better_count += 1
                else:
                    not_better_count = 0
                    best_f1 = dev_f1
                    best_f1_datasets = copy.deepcopy(temp_rlt)
                    print('best result: ', temp_rlt)
                    print('save model to', self.args.model_path)
                    self.model.save(self.args.model_path)
                    torch.save(self.model.state_dict(), self.args.model_path + '.state_dict')
                    print('finish best model \n')
                if not_better_count == self.args.n_not_inc:
                    break

        print('best_rlts', best_f1_datasets)


    def predict(self, data):
        predictions = {items[0]['doc_name']: [] for items in data}
        self.model.eval()

        for batch in data:  # each document is a minibatch
            token_ids = [m['context'][0] + m['context'][1]
                         if len(m['context'][0]) + len(m['context'][1]) > 0
                         else [self.word_vocab.unk_id]
                         for m in batch]

            entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).to(DEVICE))
            p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).to(DEVICE))
            entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).to(DEVICE))
            true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).to(DEVICE))
            mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).to(DEVICE))
            etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).to(DEVICE)) 

            token_ids, token_mask = utils.make_equal_len(token_ids, self.word_vocab.unk_id)
            token_ids = Variable(torch.LongTensor(token_ids).to(DEVICE))
            token_mask = Variable(torch.FloatTensor(token_mask).to(DEVICE))

            n_ments = len(batch)
            n_cands = self.args.keep_ctx_ent + self.args.keep_p_e_m

            ment_name = [m['ment_name'] for m in batch]
            cands_name = [m['named_cands'] for m in batch]
            name_dis = [[0 for _ in range(n_cands)] for _ in range(n_ments)]
            for i in range(n_ments):
                for j in range(n_cands):
                    name_dis[i][j] = Levenshtein.jaro_winkler(ment_name[i], cands_name[i][j])                        
            name_dis = Variable(torch.FloatTensor(name_dis).to(DEVICE)) 

            context_tokens = [ m['context_tokens'] for m in batch]
            context_tokens = np.array(context_tokens).repeat(entity_ids.shape[1]).tolist()  # shape(batch, 8, len_token)

            ment_ctx_tokens = [ ' '.join(m['ment_ctx_token']) for m in batch]
            ent_desc_tokens = [[[] for _ in range(n_cands)] for _ in range(n_ments)]
            for i in range(n_ments):
                for j in range(n_cands):
                    ent_desc_tokens[i][j] = ' '.join(self.ent_desc_token[batch[i]['selected_cands']['cands'][j]])
            ment_ctx_tokens = np.array(ment_ctx_tokens).repeat(entity_ids.shape[1]).tolist()
            ent_desc_tokens = np.array(ent_desc_tokens).reshape(-1).tolist()

            # local model : natt + tom
            scores, mention_rep, entity_rep = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, name_dis)

            # local model : bert
            # scores = self.model.forward(context_tokens, ent_desc_tokens, token_ids, entity_ids, mtype, etype, p_e_m, name_dis)  # scores shape(batch, n_cands)
            scores = scores.cpu().data.numpy()

            pred_ids = np.argmax(scores, axis=1)
            pred_entities = [m['selected_cands']['named_cands'][i] if m['selected_cands']['mask'][i] == 1
                             else (m['selected_cands']['named_cands'][0] if m['selected_cands']['mask'][0] == 1 else 'NIL')
                             for (i, m) in zip(pred_ids, batch)]
            doc_names = [m['doc_name'] for m in batch]

            for dname, entity in zip(doc_names, pred_entities):
                predictions[dname].append({'pred': (entity, 0.)})

        return predictions


    def save_source_data(self, datasets):
        n_cands = self.args.keep_ctx_ent + self.args.keep_p_e_m
        for di, (dname, data) in enumerate(datasets):
            # Each dataset is stored in a file
            with open('/home/liuyu/kesel/local_encoder/process_data/cut_candidate/' + dname + '_cut_candidate', 'a+') as cand_file:
                for batch in data: 
                    token_ids = [m['context'][0] + m['context'][1]
                                if len(m['context'][0]) + len(m['context'][1]) > 0
                                else [self.word_vocab.unk_id]
                                for m in batch]
                    token_ids, token_mask = utils.make_equal_len(token_ids, self.word_vocab.unk_id)

                    entity_ids = [m['selected_cands']['cands'] for m in batch]
                    entity_mask = [m['selected_cands']['mask'] for m in batch]  
                    entity_titles = [m['selected_cands']['named_cands'] for m in batch]
                    p_e_m = [m['selected_cands']['p_e_m'] for m in batch]
                    true_pos = [m['selected_cands']['true_pos'] for m in batch]
                    entity_ids = np.array(entity_ids).reshape(-1).tolist()  # shape(batch, n_cands) -> (batch*n_cands)
                    entity_titles = np.array(entity_titles).reshape(-1).tolist() 
                    p_e_m = np.array(p_e_m).reshape(-1).tolist()  
                    label = []  # shape(batch, 1) -> (batch*n_cands)
                    for m_pos in true_pos:
                        for i in range(n_cands) :
                            if m_pos == i:
                                label.append(1)
                            else:
                                label.append(0)
                
                    mtype = [m['mtype'] for m in batch]
                    etype = [m['selected_cands']['etype'] for m in batch]
                    mtype = np.array(mtype).repeat(n_cands, axis=0).tolist() 
                    etype = np.array(etype).reshape(-1,4).tolist()  
                    context_sent = [m['context_tokens'] for m in batch]
                    context_sent = np.array(context_sent).repeat(n_cands).tolist()

                    ment_ctx_sent = [ ' '.join(m['ment_ctx_token']) for m in batch]
                    ent_desc_sent = [[[] for j in range(len(batch[i]['selected_cands']['cands']))] for i in range(len(batch))]
                    for i in range(len(batch)):
                        for j in range(len(batch[i]['selected_cands']['cands'])):
                            ent_desc_sent[i][j] = ' '.join(self.ent_desc_token[batch[i]['selected_cands']['cands'][j]])
                    ment_ctx_sent = np.array(ment_ctx_sent).repeat(n_cands).tolist() # shape(batch,) -> (batch*n_cands,)
                    ent_desc_sent = np.array(ent_desc_sent).reshape(-1).tolist()
                    
                    ment_name = [m['ment_name'] for m in batch]
                    ment_offset_start = [m['ment_offset_start'] for m in batch]
                    ment_offset_end = [m['ment_offset_end'] for m in batch]
                    ment_name = np.array(ment_name).repeat(n_cands).tolist()  # shape(batch,) -> (batch*n_cands,)
                    ment_offset_start = np.array(ment_offset_start).repeat(n_cands).tolist()  
                    ment_offset_end = np.array(ment_offset_end).repeat(n_cands).tolist()

                    for i in range(len(batch)*n_cands):
                        # Save candidate entity file
                        cand_content = str(label[i]) + '\t' + str(ment_name[i]) + '\t' + str(ment_offset_start[i]) + '\t' + str(ment_offset_end[i]) + '\t' + \
                            str(entity_titles[i]) + '\t' + str(entity_ids[i]) + '\t' + str(p_e_m[i]) + '\t' + str(mtype[i]) + '\t' + str(etype[i]) + '\t' + \
                            str(ment_ctx_sent[i]) + '\t' + str(ent_desc_sent[i]) + '\n'
                        cand_file.write(cand_content)

    def save_local_representation(self, datasets):         
        for di, (dname, data) in enumerate(datasets):
            local_represent_list = [] 
            for batch in data: 
                token_ids = [m['context'][0] + m['context'][1]
                            if len(m['context'][0]) + len(m['context'][1]) > 0
                            else [self.model.word_voca.unk_id]
                            for m in batch]

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).to(DEVICE))
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).to(DEVICE))
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).to(DEVICE))
                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).to(DEVICE))
                mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).to(DEVICE))
                etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).to(DEVICE)) 

                token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
                token_ids = Variable(torch.LongTensor(token_ids).to(DEVICE))
                token_mask = Variable(torch.FloatTensor(token_mask).to(DEVICE))

                n_ments = len(batch)
                n_cands = self.args.keep_ctx_ent + self.args.keep_p_e_m

                ment_name = [m['ment_name'] for m in batch]
                cands_name = [m['named_cands'] for m in batch]
                name_dis = [[0 for _ in range(n_cands)] for _ in range(n_ments)]
                for i in range(n_ments):
                    for j in range(n_cands):
                        name_dis[i][j] = Levenshtein.jaro_winkler(ment_name[i], cands_name[i][j])                        
                name_dis = Variable(torch.FloatTensor(name_dis).to(DEVICE)) 
                scores, mention_rep, entity_rep = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, name_dis)  

                scores = scores.cpu().data.numpy()            # shape(batch, n_cands)
                mention_rep = mention_rep.cpu().data.numpy()  # shape(batch*n_cands, hidden_size)
                entity_rep = entity_rep.cpu().data.numpy()    # shape(batch*n_cands, hidden_size)
                for i in range(mention_rep.shape[0]):
                    mention_entity = np.vstack((mention_rep[i], entity_rep[i]))
                    local_represent_list.append(mention_entity)

            # shape=(batch, 2, local_representation_size)
            print("local_represent_list:{0}".format(len(local_represent_list)))
            local_represent_np = np.array(local_represent_list)
            np.save('/home/liuyu/kesel/local_encoder/process_data/local_rep/' + dname + '_local_rep.npy', local_represent_np) # local_represent_np 将mention和一个cand entity在一起表示

    def rank_candidate(self, datasets, is_random): 
        for di, (dname, data) in enumerate(datasets):
            pred_index_list = [] 
            for batch in data: 
                token_ids = [m['context'][0] + m['context'][1]
                            if len(m['context'][0]) + len(m['context'][1]) > 0
                            else [self.model.word_voca.unk_id]
                            for m in batch]

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).to(DEVICE))
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).to(DEVICE))
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).to(DEVICE))
                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).to(DEVICE))
                mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).to(DEVICE))
                etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).to(DEVICE)) 

                token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
                token_ids = Variable(torch.LongTensor(token_ids).to(DEVICE))
                token_mask = Variable(torch.FloatTensor(token_mask).to(DEVICE))

                n_ments = len(batch)
                n_cands = self.args.keep_ctx_ent + self.args.keep_p_e_m

                ment_name = [m['ment_name'] for m in batch]
                cands_name = [m['named_cands'] for m in batch]
                name_dis = [[0 for _ in range(n_cands)] for _ in range(n_ments)]
                for i in range(n_ments):
                    for j in range(n_cands):
                        name_dis[i][j] = Levenshtein.jaro_winkler(ment_name[i], cands_name[i][j])                        
                name_dis = Variable(torch.FloatTensor(name_dis).to(DEVICE)) 
                scores, mention_rep, entity_rep = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, name_dis)  

                scores = scores.cpu().data.numpy()
                pred_batch = np.argmax(scores, axis=1) 

                # Record the target entity index
                tmp_pred_list = []
                for index, pred_item in enumerate(pred_batch):
                    tmp_pred_list.append(pred_item)
                pred_index_list.extend(tmp_pred_list)
            
            rank_candidate_list = []
            # Put the predicted target entity first with a certain probability
            if is_random:
                for count, pos_index in enumerate(pred_index_list):
                    start_index = count * (self.args.keep_ctx_ent + self.args.keep_p_e_m)
                    tmp_list = []

                    tmp_choice = np.random.choice(2, p=[0.5, 0.5])
                    if tmp_choice == 0:
                        # Put the predicted target entity first
                        for candidate_index in range(self.args.keep_ctx_ent + self.args.keep_p_e_m):
                            if candidate_index == 0:
                                tmp_list.append(pos_index)
                            elif candidate_index != pos_index:
                                tmp_list.append(candidate_index)
                            elif candidate_index == pos_index:
                                tmp_list.append(0)
                    else:
                        # no rank
                        tmp_list.extend([candidate_index for candidate_index in range(self.args.keep_ctx_ent + self.args.keep_p_e_m)])

                    rank_candidate_list.extend([start_index+ele for ele in tmp_list])

            else:
                for count, pos_index in enumerate(pred_index_list):
                    start_index = count * (self.args.keep_ctx_ent + self.args.keep_p_e_m)
                    tmp_list = []

                    # Put the predicted target entity first
                    for candidate_index in range(self.args.keep_ctx_ent + self.args.keep_p_e_m):
                        if candidate_index == 0:
                            tmp_list.append(pos_index)
                        elif candidate_index != pos_index:
                            tmp_list.append(candidate_index)
                        elif candidate_index == pos_index:
                            tmp_list.append(0)

                    rank_candidate_list.extend([start_index + ele for ele in tmp_list])
            print("rank_candidate_list:{}".format(len(rank_candidate_list)))

            train_entity_list = []
            source_data_path = '/home/liuyu/kesel/local_encoder/process_data/cut_candidate/' + dname + '_cut_candidate'
            rank_data_path = '/home/liuyu/kesel/local_encoder/process_data/cut_candidate_rank/' + dname + '_cut_candidate_rank'
            with open(source_data_path, "r") as source_data_file:
                with open(rank_data_path, "w") as data_rank_file:
                    for item in source_data_file:
                        item = item.strip()
                        train_entity_list.append(item)

                    for rank_index in rank_candidate_list:
                        data_rank_file.write(train_entity_list[rank_index] + "\n")
                        
            source_rep_path = '/home/liuyu/kesel/local_encoder/process_data/local_rep/' + dname + '_local_rep.npy'
            local_rep_np = np.load(source_rep_path)
            local_rep_rank_list = []
            for rank_index in rank_candidate_list:
                local_rep_rank_list.append(local_rep_np[rank_index])
            print("local_rep_np:{0}, local_rep_rank_list:{1}".format(len(local_rep_np), len(local_rep_rank_list)))
            rank_rep_path = '/home/liuyu/kesel/local_encoder/process_data/local_rep_rank/' + dname + '_local_rep_candidate_rank.npy'
            np.save(rank_rep_path, np.array(local_rep_rank_list))

    def rank_mention(self, datasets, datasets_statistics):
        """
        rank mentions in a segment
        """
        rank_mention_num = self.args.rank_mention_num 
        candidate_entity_num = (self.args.keep_ctx_ent + self.args.keep_p_e_m)
        for di, (dname, data) in enumerate(datasets):
            # cnt = 0
            all_sim_list = []
            for batch in data: 
                token_ids = [m['context'][0] + m['context'][1]
                            if len(m['context'][0]) + len(m['context'][1]) > 0
                            else [self.model.word_voca.unk_id]
                            for m in batch]

                entity_ids = Variable(torch.LongTensor([m['selected_cands']['cands'] for m in batch]).to(DEVICE))
                p_e_m = Variable(torch.FloatTensor([m['selected_cands']['p_e_m'] for m in batch]).to(DEVICE))
                entity_mask = Variable(torch.FloatTensor([m['selected_cands']['mask'] for m in batch]).to(DEVICE))
                true_pos = Variable(torch.LongTensor([m['selected_cands']['true_pos'] for m in batch]).to(DEVICE))
                mtype = Variable(torch.FloatTensor([m['mtype'] for m in batch]).to(DEVICE))
                etype = Variable(torch.FloatTensor([m['selected_cands']['etype'] for m in batch]).to(DEVICE)) 

                token_ids, token_mask = utils.make_equal_len(token_ids, self.model.word_voca.unk_id)
                token_ids = Variable(torch.LongTensor(token_ids).to(DEVICE))
                token_mask = Variable(torch.FloatTensor(token_mask).to(DEVICE))

                n_ments = len(batch)
                n_cands = self.args.keep_ctx_ent + self.args.keep_p_e_m

                ment_name = [m['ment_name'] for m in batch]
                cands_name = [m['named_cands'] for m in batch]
                name_dis = [[0 for _ in range(n_cands)] for _ in range(n_ments)]
                for i in range(n_ments):
                    for j in range(n_cands):
                        name_dis[i][j] = Levenshtein.jaro_winkler(ment_name[i], cands_name[i][j])                        
                name_dis = Variable(torch.FloatTensor(name_dis).to(DEVICE)) 
                scores, mention_rep, entity_rep = self.model.forward(token_ids, token_mask, entity_ids, entity_mask, p_e_m, mtype, etype, name_dis)  

                scores = scores.cpu().data.numpy()
                max_sim = np.max(scores, axis=-1)
                all_sim_list.extend([item for item in max_sim])

            doc_index = 0
            max_sim_list = []
            rank_entity_list = []
            entity_start_index = 0
            doc_num = datasets_statistics[dname]['doc_num']
            doc_mention_num = datasets_statistics[dname]['doc_mention_num'][doc_index]
        
            for max_sim in all_sim_list:
                max_sim_list.append(max_sim)

                if len(max_sim_list) == doc_mention_num:
                    doc_group_num = int((doc_mention_num - 1) / rank_mention_num) + 1
                    for doc_group_index in range(doc_group_num):
                        start_id = doc_group_index * rank_mention_num
                        end_id = min((doc_group_index + 1) * rank_mention_num, doc_mention_num)
                        tmp_sim_list = max_sim_list[start_id:end_id]
                        tmp_dict = {}
                        for index, val in enumerate(tmp_sim_list):
                            tmp_dict[index] = val
                        rank_mention_index_list = [item[0] for item in sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True)]

                        for rank_mention_index in rank_mention_index_list:
                            rank_entity_list.extend([entity_start_index + entity_index for entity_index in
                                                    range(candidate_entity_num * rank_mention_index,
                                                        candidate_entity_num * (rank_mention_index + 1))])

                        entity_start_index += (end_id - start_id) * candidate_entity_num

                    max_sim_list = []
                    doc_index += 1
                    if doc_index < doc_num:
                        doc_mention_num = datasets_statistics[dname]['doc_mention_num'][doc_index]
          
            print("rank_entity_list:{0}, set(rank_entity_list):{1}".format(len(rank_entity_list), len(set(rank_entity_list))))


            # save the ranked data
            train_entity_list = []
            source_data_path = '/home/liuyu/kesel/local_encoder/process_data/cut_candidate_rank/' + dname + '_cut_candidate_rank'
            rank_data_path = '/home/liuyu/kesel/local_encoder/process_data/cut_candidate_mention_rank/' + dname + '_mention_rank'
            with open(source_data_path, "r") as source_data_file:
                with open(rank_data_path, "w") as data_rank_file:
                    for item in source_data_file:
                        item = item.strip()
                        train_entity_list.append(item)

                    for rank_index in rank_entity_list:
                        data_rank_file.write(train_entity_list[rank_index] + "\n")

            # save the ranked data
            source_rep_path = '/home/liuyu/kesel/local_encoder/process_data/local_rep_rank/' + dname + '_local_rep_candidate_rank.npy'
            rank_rep_path = '/home/liuyu/kesel/Blocal_encoder/process_data/local_rep_mention_rank/' + dname + '_local_rep_mention_rank.npy'
            local_rep_np = np.load(source_rep_path)
            local_rep_rank_list = []
            for rank_index in rank_entity_list:
                local_rep_rank_list.append(local_rep_np[rank_index])

            print("local_rep_np:{0}, local_rep_rank_list:{1}".format(len(local_rep_np), len(local_rep_rank_list)))
            np.save(rank_rep_path, np.array(local_rep_rank_list))

