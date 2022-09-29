import sys

sys.path.append('../keyclass/')
import torch
import argparse
from os.path import join
import utils
import models
import pickle


def run(args_cmd):
    args = utils.Parser(config_file_path=args_cmd.config).parse()

    if args['use_custom_encoder']:
        model = models.CustomEncoder(
            pretrained_model_name_or_path=args['base_encoder'],
            device='cuda' if torch.cuda.is_available() else 'cpu')
    else:
        model = models.Encoder(
            model_name=args['base_encoder'],
            device='cuda' if torch.cuda.is_available() else 'cpu')

    for split in ['train', 'test']:
        sentences = utils.fetch_data(dataset=args['dataset'],
                                     split=split,
                                     path=args['data_path'])
        embeddings = model.encode(
            sentences=sentences,
            batch_size=args['end_model_batch_size'],
            show_progress_bar=args['show_progress_bar'],
            normalize_embeddings=args['normalize_embeddings'])
        with open(
                join(args['data_path'], args['dataset'],
                     f'{split}_embeddings.pkl'), 'wb') as f:
            pickle.dump(embeddings, f)


if __name__ == "__main__":
    parser_cmd = argparse.ArgumentParser()
    parser_cmd.add_argument('--config',
                            default='../default_config.yml',
                            help='Configuration file')
    args_cmd = parser_cmd.parse_args()

    run(args_cmd)
