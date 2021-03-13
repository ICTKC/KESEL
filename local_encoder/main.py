import dataset as D
from ed_ranker import EDRanker
import utils
import argparse
import numpy as np
import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

parser = argparse.ArgumentParser()

# general args
parser.add_argument("--mode", type=str,
                    help="train or eval",
                    default='train')
parser.add_argument("--model_path", type=str,
                    help="model path to save/load",
                    default='')

# args for preranking
parser.add_argument("--n_cands_before_rank", type=int,
                    help="number of candidates",
                    default=30)
parser.add_argument("--prerank_ctx_window", type=int,
                    help="size of context window for the preranking model",
                    default=50)
parser.add_argument("--keep_p_e_m", type=int,
                    help="number of top candidates to keep w.r.t p(e|m)",
                    default=3)
parser.add_argument("--keep_ctx_ent", type=int,
                    help="number of top candidates to keep w.r.t using context",
                    default=3)

# args for local encoder
parser.add_argument("--ctx_window", type=int,
                    help="size of context window for the local model",
                    default=100)
parser.add_argument("--tok_top_n", type=int,
                    help="number of top contextual words for the local model",
                    default=50)  # 25
parser.add_argument("--hid_dims", type=int,
                    help="number of hidden neurons",
                    default=100)
parser.add_argument("--dropout_rate", type=float,
                    help="dropout rate for relation scores",
                    default=0.3)

# args for global encoder
parser.add_argument("--rank_mention_num", type=int,
                    help="number of mentions in a segment",
                    default=3) 
parser.add_argument("--flag", type=str,
                    help="flag of generating global data: T or F",
                    default='F')

# args for training
parser.add_argument("--n_epochs", type=int,
                    help="max number of epochs",
                    default=300)
parser.add_argument("--dev_f1_change_lr", type=float,
                    help="dev f1 to change learning rate",
                    default=0.915)
parser.add_argument("--n_not_inc", type=int,
                    help="number of evals after dev f1 not increase",
                    default=100)
parser.add_argument("--eval_after_n_epochs", type=int,
                    help="number of epochs to eval",
                    default=5)
parser.add_argument("--learning_rate", type=float,
                    help="learning rate",
                    default=1e-4)
parser.add_argument("--margin", type=float,
                    help="margin",
                    default=0.1)
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
args = parser.parse_args()

# if gpu is to be used
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)    # set random seed for cpu
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)   # set random seed for present GPU

datadir = '/home/liuyu/kesel/data/generated/test_train_data/'
conll_path = '/home/liuyu/kesel/data/basic_data/test_datasets/'
person_path = '/home/liuyu/kesel/data/basic_data/p_e_m_data/persons.txt'
voca_emb_dir = "/home/liuyu/kesel/data/generated/embeddings/word_ent_embs/"


if __name__ == "__main__":
    print('load conll at', datadir)
    conll = D.CoNLLDataset(datadir, person_path, conll_path)

    print('create model')
    word_voca, word_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.word',
                                                      voca_emb_dir + 'word_embeddings.npy')
    entity_voca, entity_embeddings = utils.load_voca_embs(voca_emb_dir + 'dict.entity',
                                                          voca_emb_dir + 'entity_embeddings.npy')

    config = {'hid_dims': args.hid_dims,
              'emb_dims': entity_embeddings.shape[1],
              'freeze_embs': True,
              'tok_top_n': args.tok_top_n,
              'margin': args.margin,
              'word_voca': word_voca,
              'entity_voca': entity_voca,
              'word_embeddings': word_embeddings,
              'entity_embeddings': entity_embeddings,
              'dr': args.dropout_rate,
              'args': args}
    ranker = EDRanker(config=config)
    dev_datasets = [
                    # ('aida-train', conll.train),
                    ('aida-A', conll.testA),
                    ('aida-B', conll.testB),
                    ('msnbc', conll.msnbc),
                    ('aquaint', conll.aquaint),
                    ('ace2004', conll.ace2004),
                    ('clueweb', conll.clueweb),
                    ('wikipedia', conll.wikipedia)
                    ]

    if args.mode == 'train':
        print('training...')
        config = {'lr': args.learning_rate, 'n_epochs': args.n_epochs}
        ranker.train(conll.train, dev_datasets, config)

    elif args.mode == 'eval':
        org_dev_datasets = dev_datasets  
        dev_datasets = []
        datasets_statistics = {}
        for dname, data in org_dev_datasets:
            dev_datasets.append((dname, ranker.get_data_items(data, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))
    
    # utilize local encoder to generate data for global encoder 
    if args.mode == 'eval' and args.flag == 'T': 
        for dname, data in org_dev_datasets:  # + [('aida-train', conll.train)]
            dev_datasets.append((dname, ranker.get_data_items(data, predict=True)))
            print(dname, '#dev docs', len(dev_datasets[-1][1]))

            doc_num = len(dev_datasets[-1][1])
            doc_mention_num_list = []
            for doc_content in dev_datasets[-1][1]:
                doc_mention_num_list.append(len(doc_content))
            datasets_statistics[dname] = {'doc_num':doc_num, 'doc_mention_num':doc_mention_num_list}
        # print(datasets_statistics)
        ranker.save_source_data(dev_datasets)
        ranker.save_local_representation(dev_datasets)
        ranker.rank_candidate(dev_datasets, False) # is_random=False
        ranker.rank_mention(dev_datasets, datasets_statistics)
