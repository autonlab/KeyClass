import sys
sys.path.append('../keyclass/')

import argparse
from os.path import join
from utils import fetch_data
from models import Encoder
import torch
import pickle
from config import Parser

parser_cmd = argparse.ArgumentParser()
parser_cmd.add_argument('--config', default='../default_config.yml', help='Configuration file')
args_cmd = parser_cmd.parse_args()

parser = Parser(config_file_path=args_cmd.config)
args = parser.parse()

model = Encoder(model_name=args['base_encoder'], device='cuda' if torch.cuda.is_available() else 'cpu')

for split in ['train', 'test']:
    sentences = fetch_data(dataset=args['dataset'], split=split, path=args['data_path'])
    embeddings = model.encode(sentences=sentences, batch_size=args['batch_size'], 
                              show_progress_bar=args['show_progress_bar'], normalize_embeddings=args['normalize_embeddings'])
    with open(join(args['data_path'], args['dataset'], f'{split}_embeddings.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
