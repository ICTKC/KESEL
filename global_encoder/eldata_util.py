# coding:utf-8

import json
import numpy as np
import torch


from knowledge_bert import BertTokenizer, BertModel, BertForMaskedLM
tokenizer = BertTokenizer.from_pretrained('/home/liuyu/kesel/global_encoder/ernie_pretrain/ernie_base')

'''
path = /home/liuyu/kesel/local_encoder/process_data/mention_rank_wiki
target:  mention_rank_wiki is processed as Ernie model input format
three mentions form a sequence, text_seq and ents_seq format is as follows:
text_a, ents_a = "Jim Henson was a puppeteer .", [['Q191037', 0, 10, 0.0], ['Q2629392', 17, 26, 0.0]]
labels_seq shape(batch, seq)  | ents_index shape(batch, seq) | cands_ents_id shape(batch, seq, n_cands)

'''
class DataUtil():
    def init(self):
        pass

    def process_global_data(self, dname, data_path, local_rep_path, group_path, local_fea_path, seq_len, candidate_entity_num):
        """
        :param data_path: 
        :param local_rep_path: local context represents
        :param group_path: mention按文档划分的group
        :param seq_len: 文档中mention序列长度
        :param candidate_entity_num: 每个mention对应候选实体数量
        :return: mention(numpy), entity(numpy), label(numpy), feature(numpy), entity_url(list)
        """

        datasets_statistics= json.load(open(group_path))

        entity_start_index = 0
        padding_index_list = []
        for doc_mention_num in datasets_statistics[dname]['doc_mention_num']:
            doc_group_num = int((doc_mention_num - 1) / seq_len) + 1
            # 对一个group内的mention进行padding
            for doc_group_index in range(doc_group_num):
                start_id = doc_group_index * seq_len
                end_id = min((doc_group_index + 1) * seq_len, doc_mention_num)
                # 对group内最后一个mention进行复制padding
                if end_id - start_id < seq_len:
                    tmp_pad_list = [entity_start_index + tmp_pad for tmp_pad in
                                    range((end_id - start_id - 1) * candidate_entity_num, (end_id - start_id) * candidate_entity_num)]
                    for time in range(seq_len-(end_id - start_id)):
                        padding_index_list.append(tmp_pad_list)

                entity_start_index += (end_id - start_id) * candidate_entity_num

        # 正例为1, 负例为0
        label_list = []
        ment_name_list = []
        ment_sent_list = []
        ment_offset_list = []
        ent_id_list = []
        p_e_m_list = []
        mtype_list = []
        etype_list = []

        with open(data_path, "r") as data_file:
            for item in data_file:
                item = item.strip().split('\t')
                label = item[0]
                ment_name, ment_start_token, ment_end_token = item[1:4]
                ent_title, ent_id = item[4:6]
                p_e_m, mtype, etype = item[6:9]
                ment_sent = item[9]
                
                ment_ctx_token = ment_sent.split(' ')
                ment_start_sent =sum([len(token) for token in  ment_ctx_token[0:int(ment_start_token)]]) + int(ment_start_token)
                ment_end_sent = ment_start_sent + len(ment_name)

                label_list.append(int(label))
                ment_name_list.append(ment_name)
                ment_sent_list.append(ment_sent)
                ment_offset_list.append((ment_start_sent, ment_end_sent))
                ent_id_list.append(ent_id)
                p_e_m_list.append(float(p_e_m))
                mtype_list.append([int(i) for i in mtype[1:-1].split(', ')])
                etype_list.append([int(i) for i in etype[1:-1].split(', ')])              

        # 加载mention和entity本地表示
        local_rep_np = np.load(local_rep_path)
        mention_rep_list = []
        entity_rep_list = []
        for mention_entity_rep in local_rep_np:
            mention_rep_list.append(mention_entity_rep[0])
            entity_rep_list.append(mention_entity_rep[1])
        
        local_fea_np = np.load(local_fea_path)
        local_fea_list = local_fea_np.tolist()

        pad_mention_rep_list = []
        pad_entity_rep_list = []
        pad_local_fea_list = []
        pad_label_list = []
        pad_ment_name_list = []
        pad_ment_sent_list = []
        pad_ment_offset_list = []
        pad_ent_id_list = []
        pad_p_e_m_list = []
        pad_mtype_list = []
        pad_etype_list = []

        # 对mention,entity,feature,entity_embedd,label进行padding填充
        last_first = 0
        for pad_indexs in padding_index_list:
            first_index = pad_indexs[0]
            if last_first == first_index:
                pad_mention_rep_list.extend(mention_rep_list[first_index:pad_indexs[-1]+1])
                pad_entity_rep_list.extend(entity_rep_list[first_index:pad_indexs[-1]+1])      
                pad_local_fea_list.extend(local_fea_list[first_index:pad_indexs[-1]+1])         
                pad_label_list.extend(label_list[first_index:pad_indexs[-1]+1])
                pad_ment_name_list.extend(ment_name_list[first_index:pad_indexs[-1]+1])
                pad_ment_sent_list.extend(ment_sent_list[first_index:pad_indexs[-1]+1])
                pad_ment_offset_list.extend(ment_offset_list[first_index:pad_indexs[-1]+1])
                pad_ent_id_list.extend(ent_id_list[first_index:pad_indexs[-1]+1])                
                pad_p_e_m_list.extend(p_e_m_list[first_index:pad_indexs[-1]+1])
                pad_mtype_list.extend(mtype_list[first_index:pad_indexs[-1]+1])
                pad_etype_list.extend(etype_list[first_index:pad_indexs[-1]+1])  
            else:
                pad_mention_rep_list.extend(mention_rep_list[last_first:first_index])
                pad_mention_rep_list.extend(mention_rep_list[first_index:pad_indexs[-1]+1])

                pad_entity_rep_list.extend(entity_rep_list[last_first:first_index])
                pad_entity_rep_list.extend(entity_rep_list[first_index:pad_indexs[-1]+1])

                pad_local_fea_list.extend(local_fea_list[last_first:first_index])
                pad_local_fea_list.extend(local_fea_list[first_index:pad_indexs[-1]+1])

                pad_label_list.extend(label_list[last_first:first_index])
                pad_label_list.extend(label_list[first_index:pad_indexs[-1]+1])

                pad_ment_name_list.extend(ment_name_list[last_first:first_index]) 
                pad_ment_name_list.extend(ment_name_list[first_index:pad_indexs[-1]+1])

                pad_ment_sent_list.extend(ment_sent_list[last_first:first_index])
                pad_ment_sent_list.extend(ment_sent_list[first_index:pad_indexs[-1]+1])

                pad_ment_offset_list.extend(ment_offset_list[last_first:first_index])
                pad_ment_offset_list.extend(ment_offset_list[first_index:pad_indexs[-1]+1])

                pad_ent_id_list.extend(ent_id_list[last_first:first_index])
                pad_ent_id_list.extend(ent_id_list[first_index:pad_indexs[-1]+1])     

                pad_p_e_m_list.extend(p_e_m_list[last_first:first_index])
                pad_p_e_m_list.extend(p_e_m_list[first_index:pad_indexs[-1]+1])

                pad_mtype_list.extend(mtype_list[last_first:first_index])
                pad_mtype_list.extend(mtype_list[first_index:pad_indexs[-1]+1])

                pad_etype_list.extend(etype_list[last_first:first_index])
                pad_etype_list.extend(etype_list[first_index:pad_indexs[-1]+1])           
            last_first = first_index

        # 添加末尾
        pad_mention_rep_list.extend(mention_rep_list[last_first:])
        pad_entity_rep_list.extend(entity_rep_list[last_first:])
        pad_local_fea_list.extend(local_fea_list[last_first:])
        pad_label_list.extend(label_list[last_first:])
        pad_ment_name_list.extend(ment_name_list[last_first:])
        pad_ment_sent_list.extend(ment_sent_list[last_first:])
        pad_ment_offset_list.extend(ment_offset_list[last_first:])
        pad_ent_id_list.extend(ent_id_list[last_first:])
        pad_p_e_m_list.extend(p_e_m_list[last_first:])
        pad_mtype_list.extend(mtype_list[last_first:])
        pad_etype_list.extend(etype_list[last_first:])

        return np.array(pad_mention_rep_list), np.array(pad_entity_rep_list), np.array(pad_local_fea_list), \
            np.array(pad_ment_name_list), np.array(pad_ment_sent_list), \
            np.array(pad_ment_offset_list), np.array(pad_ent_id_list), \
            np.array(pad_mtype_list), np.array(pad_etype_list), \
            np.array(pad_p_e_m_list), np.array(pad_label_list)   # pad_label_list


    def get_global_feature_input(self, ment_names,  ment_sents, ment_offsets, ent_ids, labels, seq_len, candidate_entity_num, batch_size=64, max_seq_len=256):

        """
        text_a, ents_a = "Jim Henson was a puppeteer .", [['Q191037', 0, 10, 0.0], ['Q2629392', 17, 26, 0.0]]
        labels_seq shape(batch, seq) 
        ents_index shape(batch, seq)
        cands_ents_id shape(batch, seq, n_cands)
        """
        # # Convert ents
        entity2id = {}
        with open("/home/liuyu/kesel/global_encoder/kg_embed/entity2id.txt") as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                entity2id[qid] = int(eid)

        # 批次数
        data_len = len(ment_names)
        num_batch = int((data_len - 1) / (batch_size * seq_len * candidate_entity_num)) + 1

        # 所有mention表示
        ment_sents_list = [ment_sents[index] for index in range(data_len) if index % candidate_entity_num == 0]
        ment_offsets_list = [ment_offsets[index] for index in range(data_len) if index % candidate_entity_num == 0]

        # # 方式1  mention按照序列进行划分
        # tmp_ment_sent = ''
        # tmp_ment_offset_list = []
        # ment_sent_seq_list = []
        # ment_offset_seq_list = []
        # for index in range(len(ment_sents_list)):
        #     offset_start_seq  = len(tmp_ment_sent)+ment_offsets_list[index][0]
        #     offset_end_seq  = len(tmp_ment_sent)+ment_offsets_list[index][1]
        #     tmp_ment_offset_list.append(['Q0',offset_start_seq, offset_end_seq, 0.0])
        #     tmp_ment_sent += ment_sents_list[index] + ' '
        #     if (index+1) % seq_len == 0:
        #         ment_sent_seq_list.append(tmp_ment_sent)
        #         ment_offset_seq_list.append(tmp_ment_offset_list) 
        #         tmp_ment_sent = ''
        #         tmp_ment_offset_list = []         

        # # 方式2  所有metion的目标实体 ent_id 和 label
        entity_count = 0
        tmp_label_list = []
        tmp_ent_id_list = []
        mention_entity_label_list = []
        mention_entity_ent_id_list = []
        for ent_id, label in zip(ent_ids, labels):
            entity_count += 1
            tmp_label_list.append(label)
            tmp_ent_id_list.append(ent_id)
            if entity_count % candidate_entity_num == 0:
                mention_entity_label_list.append(tmp_label_list)
                mention_entity_ent_id_list.append(tmp_ent_id_list)
                tmp_label_list = []
                tmp_ent_id_list = []

        # 候选实体按照序列进行划分
        mention_label_list = []
        mention_ent_id_list = []
        cnt_notin = 0
        for mention_entity_ent_id, mention_entity_label in zip(mention_entity_ent_id_list,mention_entity_label_list):
            target_label_index = np.argmax(mention_entity_label)
            mention_label_list.append(target_label_index) 
            mention_ent_id_list.append(mention_entity_ent_id[target_label_index])
            if mention_entity_ent_id[target_label_index] not in entity2id:
                cnt_notin += 1
            
        # print("ment_sents num: {0}, ment_offsets num: {1}, mention_ent_id num: {2}, mention_label num: {3}" \
        #     .format(len(ment_sents_list), len(ment_offsets_list), len(mention_ent_id_list), len(mention_label_list)))    

        # mention按照序列进行划分
        tmp_ment_sent = ''
        tmp_ment_offset_list = []
        tmp_ment_label_list = []
        seq_ment_sent_list = []
        seq_ment_offset_list = []
        seq_ment_label_list = []

        for index in range(len(ment_sents_list)):
            target_ent_id = mention_ent_id_list[index]  # 当未知时，设置为Q0
            offset_start_seq  = len(tmp_ment_sent)+ment_offsets_list[index][0]
            offset_end_seq  = len(tmp_ment_sent)+ment_offsets_list[index][1]
            tmp_ment_offset_list.append([target_ent_id, offset_start_seq, offset_end_seq, 0.0])
            tmp_ment_sent += ment_sents_list[index] + ' '

            tmp_ment_label_list.append(mention_label_list[index])
            if (index+1) % seq_len == 0:
                if len(tmp_ment_offset_list)!=seq_len or len(tmp_ment_label_list)!=seq_len:
                    print(tmp_ment_offset_list, tmp_ment_label_list)
                seq_ment_sent_list.append(tmp_ment_sent)
                seq_ment_offset_list.append(tmp_ment_offset_list) 
                seq_ment_label_list.append(tmp_ment_label_list)
                tmp_ment_sent = ''
                tmp_ment_offset_list = [] 
                tmp_ment_label_list = []
        # print("seq_ment_sent num: {0}, seq_ment_offset num: {1}, seq_ment_label num: {2}" \
        #     .format(len(seq_ment_sent_list), len(seq_ment_offset_list), len(seq_ment_label_list)))    

        # Tokenize
        seq_tokens_list = []
        seq_tokens_mask_list = []
        seq_tokens_segment_list = []
        seq_ents_list = []
        seq_ents_mask_list = []
        seq_ents_index_list = []
        res_1, res_2, res_3 = 0, 0, 0
        for text_seq, ent_seq in zip(seq_ment_sent_list, seq_ment_offset_list):
            token_seq, entity_seq = tokenizer.tokenize(text_seq, ent_seq)              
            token_seq = ["[CLS]"] + token_seq + ["[SEP]"]
            entity_seq = ["UNK"] + entity_seq + ["UNK"]
            token_mask_seq = [1] * len(token_seq)
            token_segment_seq = [0] * len(token_seq)
            

            indexed_token_seq = tokenizer.convert_tokens_to_ids(token_seq) 
            indexed_ent_seq = []
            ent_mask_seq = []
            ent_index_seq = []
            for index, ent in enumerate(entity_seq):
                if ent != "UNK":
                    if ent in entity2id:
                        indexed_ent_seq.append(entity2id[ent])
                    else:
                        indexed_ent_seq.append(-1)
                    ent_mask_seq.append(1)
                    ent_index_seq.append(index)
                else:
                    indexed_ent_seq.append(-1)
                    ent_mask_seq.append(0)
            ent_mask_seq[0] = 1

            # Zero-pad up to the sequence length.
            if len(indexed_token_seq) >= max_seq_len:
                for i in range(-1, -1*(seq_len+1), -1):
                    if ent_index_seq[i] >= max_seq_len:
                        indexed_token_seq[max_seq_len+i] = indexed_token_seq[ent_index_seq[i]]
                        indexed_ent_seq[max_seq_len+i] = indexed_ent_seq[ent_index_seq[i]]
                        ent_mask_seq[max_seq_len+i] = ent_mask_seq[ent_index_seq[i]]
                        ent_index_seq[i] = max_seq_len+i   
                    else:
                        break                      

                indexed_token_seq = indexed_token_seq[:max_seq_len]
                token_mask_seq = token_mask_seq[:max_seq_len]
                token_segment_seq = token_segment_seq[:max_seq_len]
                indexed_ent_seq = indexed_ent_seq[:max_seq_len]
                ent_mask_seq = ent_mask_seq[:max_seq_len]
            else:
                padding = [0] * (max_seq_len - len(indexed_token_seq))
                padding_ = [-1] * (max_seq_len - len(indexed_token_seq))
                indexed_token_seq += padding
                token_mask_seq += padding
                token_segment_seq += padding   
                indexed_ent_seq += padding_
                ent_mask_seq += padding
                
            assert len(indexed_token_seq) == max_seq_len
            assert len(token_mask_seq) == max_seq_len
            assert len(token_segment_seq) == max_seq_len
            assert len(indexed_ent_seq) == max_seq_len
            assert len(ent_mask_seq) == max_seq_len

            seq_tokens_list.append(indexed_token_seq)
            seq_tokens_mask_list.append(token_mask_seq)
            seq_tokens_segment_list.append(token_segment_seq)
            seq_ents_list.append(indexed_ent_seq)
            seq_ents_mask_list.append(ent_mask_seq)
            seq_ents_index_list.append(ent_index_seq)

        print("seq_tokens num: {0}, seq_ents num: {1}, seq_ents_index num: {2}" \
            .format(len(seq_tokens_list), len(seq_ents_list), len(seq_ents_index_list)))    
        # print('res_1: {}, res_2: {}, res_3:{}'.format(res_1, res_2, res_3))
        # print("target entity not in entity2id num:{}".format(cnt_notin))

        seq_tokens_np = np.array(seq_tokens_list)  # shape(97,ctxlen)
        seq_tokens_mask_np = np.array(seq_tokens_mask_list)   #shape(97,ctxlen)
        seq_tokens_segment_np = np.array(seq_tokens_segment_list)  #shape(97,ctxlen)
        seq_ents_np = np.array(seq_ents_list)  #shape(97,ctxlen)
        seq_ents_mask_np = np.array(seq_ents_mask_list)  #shape(97,3)
        seq_ents_index_np = np.array(seq_ents_index_list)
        seq_label_np = np.array(seq_ment_label_list)
        # print(seq_label_np[0])

        # for i in range(num_batch):
        #     start_id = i * batch_size
        #     end_id = min((i + 1) * batch_size, data_len / (seq_len * candidate_entity_num))
            # yield seq_tokens_np[start_id:end_id], seq_tokens_mask_np[start_id:end_id], seq_tokens_segment_np[start_id:end_id], \
            #   seq_ents_np[start_id:end_id], seq_ents_mask_np[start_id:end_id], seq_ents_index_np[start_id:end_id], \
            #   seq_label_np[start_id:end_id]
        return seq_tokens_np, seq_tokens_mask_np, seq_tokens_segment_np, seq_ents_np, seq_ents_mask_np, seq_ents_index_np, seq_label_np


    def get_local_feature_input(self, mentions, entities, localfeas, entids, mtypes, etypes, pems, labels, seq_len, candidate_entity_num, batch_size=32):
        # # Convert ents
        entity2id = {}
        with open("/home/liuyu/kesel/global_encoder/kg_embed/entity2id.txt") as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                entity2id[qid] = int(eid)        
        
        # 批次数
        data_len = len(mentions)
        num_batch = int((data_len - 1) / (batch_size * seq_len * candidate_entity_num)) + 1

        # 所有mention表示
        mention_list = [mentions[index] for index in range(data_len) if index % candidate_entity_num == 0]
        
        # mention按照序列进行划分
        mention_count = 0
        tmp_mention_list = []
        mention_seq_list = []
        for mention in mention_list:
            mention_count += 1
            tmp_mention_list.append(mention)
            if mention_count % seq_len == 0:
                mention_seq_list.append(tmp_mention_list)
                tmp_mention_list = []

        # 候选实体按照mention划分
        entity_count = 0
        tmp_entity_list = []
        tmp_localfea_list = []
        tmp_pem_list = []
        tmp_mtype_list = []
        tmp_etype_list = []
        tmp_label_list = []
        tmp_entid_list = []
        mention_entity_list = []
        mention_entity_localfea_list = []
        mention_entity_pem_list = []
        mention_entity_mtype_list = []
        mention_entity_etype_list = []
        mention_entity_label_list = []
        mention_entity_entid_list = []
        cnt_notin = 0
        for entity_rep, localfea, ent_id, pem, mtype, etype, label in zip(entities, localfeas, entids, pems, mtypes, etypes, labels):

            entity_count += 1
            tmp_entity_list.append(entity_rep)
            tmp_localfea_list.append(localfea)
            tmp_pem_list.append(pem)
            tmp_mtype_list.append(mtype)
            tmp_etype_list.append(etype)
            tmp_label_list.append(label)
            if ent_id not in entity2id:
                tmp_entid_list.append(-1)
                cnt_notin += 1
            else:
                tmp_entid_list.append(entity2id[ent_id])  # 将qid转为eid，为了在转换数据类型时方便 torch.tensor

            if entity_count % candidate_entity_num == 0:
                mention_entity_list.append(tmp_entity_list)
                mention_entity_localfea_list.append(tmp_localfea_list)
                mention_entity_pem_list.append(tmp_pem_list)
                mention_entity_mtype_list.append(tmp_mtype_list)
                mention_entity_etype_list.append(tmp_etype_list)
                mention_entity_label_list.append(tmp_label_list)
                mention_entity_entid_list.append(tmp_entid_list)
                tmp_entity_list = []
                tmp_localfea_list = []
                tmp_pem_list = []
                tmp_mtype_list = []
                tmp_etype_list = []
                tmp_label_list = []
                tmp_entid_list = []

        # 候选实体按照序列进行划分
        mention_count = 0
        tmp_entity_list = []
        tmp_localfea_list = []
        tmp_pem_list = []
        tmp_mtype_list = []
        tmp_etype_list = []
        tmp_label_list = []
        tmp_entid_list = []
        entity_seq_list = []
        localfea_seq_list = []
        pem_seq_list = []
        mtype_seq_list = []
        etype_seq_list = []
        label_seq_list = []
        entid_seq_list = []
        for mention_entity, mention_entity_localfea, mention_entity_entid, mention_entity_pem, mention_entity_mtype, mention_entity_etype, mention_entity_label in \
            zip(mention_entity_list, mention_entity_localfea_list, mention_entity_entid_list, mention_entity_pem_list, mention_entity_mtype_list, mention_entity_etype_list, mention_entity_label_list):

            mention_count += 1
            tmp_entity_list.append(mention_entity)
            tmp_localfea_list.append(mention_entity_localfea)
            tmp_pem_list.append(mention_entity_pem)
            tmp_mtype_list.append(mention_entity_mtype)
            tmp_etype_list.append(mention_entity_etype)
            tmp_label_list.append(mention_entity_label)
            tmp_entid_list.append(mention_entity_entid)

            if mention_count % seq_len == 0:
                entity_seq_list.append(tmp_entity_list)
                localfea_seq_list.append(tmp_localfea_list)
                pem_seq_list.append(tmp_pem_list)
                mtype_seq_list.append(tmp_mtype_list)
                etype_seq_list.append(tmp_etype_list)
                label_seq_list.append(tmp_label_list)
                entid_seq_list.append(tmp_entid_list)
                tmp_entity_list = []
                tmp_localfea_list = []
                tmp_pem_list = []
                tmp_mtype_list = []
                tmp_etype_list = []
                tmp_label_list = []
                tmp_entid_list = []

        mention_seq_np = np.array(mention_seq_list)  # shape(97,3,768)
        entity_seq_np = np.array(entity_seq_list)  # shape(97,3,6,768)
        local_fea_np = np.array(localfea_seq_list) # shape(97,3,6)
        pem_seq_np = np.array(pem_seq_list)    # shape(97,3,6)
        mtype_seq_np = np.array(mtype_seq_list)    # shape(97,3,6,4)
        etype_seq_np = np.array(etype_seq_list)    # shape(97,3,6,4)
        label_seq_np = np.array(label_seq_list)  # shape(97,3,6) 
        entid_seq_np = np.array(entid_seq_list)  # shape(97,3,6)

        # print("batch seq num:{0}, {1}, {2}"\
        #     .format(len(mention_seq_np), len(entity_seq_list), len(label_seq_np)))
        # print("cand entity not in entity2id num:{}".format(cnt_notin))
        # for i in range(num_batch):
        #     start_id = i * batch_size
        #     end_id = min((i + 1) * batch_size, data_len / (seq_len * candidate_entity_num))
            # yield mention_seq_np[start_id:end_id], entity_seq_np[start_id:end_id], \
            # entid_seq_np[start_id:end_id], pem_seq_np[start_id:end_id], \
            # mtype_seq_np[start_id:end_id], etype_seq_np[start_id:end_id], label_seq_np[start_id:end_id]    # shape=(B,4,H) 对吗？

        return mention_seq_np, entity_seq_np, local_fea_np, entid_seq_np, pem_seq_np, mtype_seq_np, etype_seq_np, label_seq_np



# if __name__ == "__main__":
#     data_obj = DataUtil()
    
#     dname_list = ['aida-train', 'aida-A', 'aida-B', 'msnbc', 'aquaint', 'ace2004', 'clueweb', 'wikipedia']
#     for dname in dname_list:
#         dname = 'ace2004'
#         # print('-'*20)
#         # print(dname)
#         data_path = '/home/liuyu/kesel/local_encoder/process_data/mention_rank_wiki/'+dname+'_mention_rank'
#         local_rep_path = '/home/liuyu/kesel/local_encoder/process_data/local_rep_mention_rank/'+dname+'_local_rep_mention_rank.npy'
#         group_path = '/home/liuyu/kesel/local_encoder/process_data/datasets_statistics.json'
#         local_fea_path = '/home/liuyu/kesel/local_encoder/process_data/local_rep_mention_rank/'+dname+'_local_fea_mention_rank.npy'
        
#         seq_len, candidate_entity_num = 3, 6
#         mentions, entities, local_feas, ment_names, ment_sents, ment_offsets, ent_ids, mtypes, etypes, pems, labels = \
#             data_obj.process_global_data(dname, data_path, local_rep_path, group_path, local_fea_path, seq_len, candidate_entity_num)

#         mention_seq_np, entity_seq_np, local_fea_np, entid_seq_np, pem_seq_np, mtype_seq_np, etype_seq_np, label_seq_np = \
#             data_obj.get_local_feature_input(mentions, entities, local_feas, ent_ids, mtypes, etypes, pems, labels, seq_len, candidate_entity_num)
    
#         seq_tokens_np, seq_tokens_mask_np, seq_tokens_segment_np, seq_ents_np, seq_ents_mask_np, seq_ents_index_np, seq_label_np = \
#             data_obj.get_global_feature_input(ment_names, ment_sents, ment_offsets, ent_ids, labels, seq_len, candidate_entity_num)
        
#         # print('--------')
#         break








