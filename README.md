# KESEL
Knowledge Enhanced Sequential Entity Linking

## Introduction
> Knowledge Enhanced Sequential Entity Linking, a novel method which converts
global entity linking into a sequence decision problem and applies a
pre-trained language model to better fuse entity knowledge.

## Download Datasets & Pre-trained Models
Download datasets from here (https://drive.google.com/open?id=1MfjzjZH_KKsXshtepzSBwkvjabdEytzh) and put datasets to the main folder (i.e. your_path/kesel)

Download BERT model from here (https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz) and ERNIE model from here (https://github.com/thunlp/ERNIE), then put models to the floder (i.e. your_path/kesel/xxx_encoder/)

## Running
### local model
1. training
cd your_path/kesel/local_encoder/
python main.py --mode train --model_path model --n_epochs 300

2. testing
python main.py --mode eval --model_path model

3. saving
python main.py --mode eval --model_path model --flag T
note that we utilize local encoder to generate data for global encoder

### global model
1. training
cd your_path/kesel/global_endoer/
python run_el.py --model_path model --seq_len 3

2. testing
python run_el.py --do_train False --do_eval True --model_path model
