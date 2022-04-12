import sys
sys.path.append('../keyclass/')

import argparse
from os.path import join
from utils import fetch_data
from models import Encoder
import torch
import pickle
from config import Parser

parser = Parser(config_file_path='../default_config.yml')
args = parser.parse()

model = Encoder(model_name=args['base_encoder'], device='cuda' if torch.cuda.is_available() else 'cpu')

for split in ['train', 'test']:
    text = fetch_data(dataset=args['dataset'], split=split, path=args['data_path'])
    embeddings = model.get_embeddings(text=text, batch_size=128)
    with open(join(args['data_path'], args['dataset'], f'{split}_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
