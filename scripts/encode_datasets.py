# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
