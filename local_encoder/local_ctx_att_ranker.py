import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from abstract_word_entity import AbstractWordEntity


class LocalCtxAttRanker(AbstractWordEntity):
    """
    local model with context token attention (from G&H's EMNLP paper)
    """

    def __init__(self, config):
        config['word_embeddings_class'] = nn.Embedding
        config['entity_embeddings_class'] = nn.Embedding
        super(LocalCtxAttRanker, self).__init__(config)

        self.tok_top_n = config['tok_top_n']
        self.att_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        self.tok_score_mat_diag = nn.Parameter(torch.ones(self.emb_dims))
        self.local_ctx_dr = nn.Dropout(p=0)
        self.hid_dims = config['hid_dims']
        self.dr = config['dr']
        # mlp
        self.score_combine = torch.nn.Sequential(
                torch.nn.Linear(20, self.hid_dims),
                torch.nn.Softplus(),
                torch.nn.Dropout(p=self.dr),
                torch.nn.Linear(self.hid_dims, 1))
        # typing feature
        self.type_matrix = torch.nn.Parameter(torch.randn([4, 5]))

    def forward(self, token_ids, tok_mask, entity_ids, entity_mask, p_e_m, mtype, etype, name_dis):
        n_ments, n_words = token_ids.size()
        n_cands = entity_ids.size(1)
        tok_mask = tok_mask.view(n_ments, 1, -1)

        tok_vecs = self.word_embeddings(token_ids)
        entity_vecs = self.entity_embeddings(entity_ids)

        ent_tok_att_scores = torch.bmm(entity_vecs * self.att_mat_diag, tok_vecs.permute(0, 2, 1))
        ent_tok_att_scores = (ent_tok_att_scores * tok_mask).add_((tok_mask - 1).mul_(1e10))
        tok_att_scores, _ = torch.max(ent_tok_att_scores, dim=1)
        top_tok_att_scores, top_tok_att_ids = torch.topk(tok_att_scores, dim=1, k=min(self.tok_top_n, n_words))
        att_probs = F.softmax(top_tok_att_scores, dim=1).view(n_ments, -1, 1)
        att_probs = att_probs / torch.sum(att_probs, dim=1, keepdim=True)

        selected_tok_vecs = torch.gather(tok_vecs, dim=1,
                                         index=top_tok_att_ids.view(n_ments, -1, 1).repeat(1, 1, tok_vecs.size(2)))
        ctx_vecs = torch.sum((selected_tok_vecs * self.tok_score_mat_diag) * att_probs, dim=1, keepdim=True)
        ctx_vecs = self.local_ctx_dr(ctx_vecs)
        ent_ctx_scores = torch.bmm(entity_vecs, ctx_vecs.permute(0, 2, 1)).view(n_ments, n_cands)        

        local_ent_scores = (ent_ctx_scores * entity_mask).add_((entity_mask - 1).mul_(1e10))
        # return scores

        mt_emb = torch.matmul(mtype, self.type_matrix).view(n_ments, 1, -1)  # shape(b,1,5)
        et_emb = torch.matmul(etype.view(-1, 4), self.type_matrix).view(n_ments, n_cands, -1) # shape(b,8,5)
        tm = torch.sum(mt_emb*et_emb, -1, True)  # shape(b,8)

        inputs = torch.cat([local_ent_scores.view(n_ments * n_cands, -1),
                            torch.log(p_e_m + 1e-20).view(n_ments * n_cands, -1),
                            tm.view(n_ments * n_cands, -1),
                            name_dis.view(n_ments * n_cands, -1)
                            ], dim=1)  # shape(b,4)
        inputs = inputs.repeat(1,5)  # shape(b,20)
        scores = self.score_combine(inputs).view(n_ments, n_cands)
        return scores, ctx_vecs.expand(-1,n_cands,-1).reshape(n_ments*n_cands,-1), entity_vecs.reshape(n_ments*n_cands,-1)  # (b,n_cands, dims)
