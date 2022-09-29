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

import argparse
import numpy as np
import torch
import os
from os.path import join, exists
import models
import utils
import train_classifier
import pickle
from datetime import datetime


def load_data(args):
    with open(
            join(args['preds_path'], f"{args['label_model']}_proba_preds.pkl"),
            'rb') as f:
        proba_preds = pickle.load(f)
    y_train_lm = np.argmax(proba_preds, axis=1)
    sample_weights = np.max(proba_preds,
                            axis=1)  # Sample weights for noise aware loss

    # Keep only very confident predictions
    mask = utils.get_balanced_data_mask(proba_preds,
                                        max_num=args['max_num'],
                                        class_balance=None)

    # Load training and testing data
    # We have already encode the dataset, so we'll just load the embeddings
    with open(
            join(args['data_path'], args['dataset'], f'train_embeddings.pkl'),
            'rb') as f:
        X_train_embed = pickle.load(f)
    with open(join(args['data_path'], args['dataset'], f'test_embeddings.pkl'),
              'rb') as f:
        X_test_embed = pickle.load(f)

    # Load training and testing ground truth labels
    training_labels_present = False
    if exists(join(args['data_path'], args['dataset'], 'train_labels.txt')):
        with open(
                join(args['data_path'], args['dataset'], f'train_labels.txt'),
                'r') as f:
            y_train = f.readlines()
        y_train = np.array([int(i.replace('\n', '')) for i in y_train])
        training_labels_present = True
    else:
        y_train = None
        print('No training labels found!')

    with open(join(args['data_path'], args['dataset'], f'test_labels.txt'),
              'r') as f:
        y_test = f.readlines()
    y_test = np.array([int(i.replace('\n', '')) for i in y_test])

    # Print data statistics
    print('\n==== Data statistics ====')
    print(
        f'Size of training data: {X_train_embed.shape}, testing data: {X_test_embed.shape}'
    )
    print(f'Size of testing labels: {y_test.shape}')
    if training_labels_present:
        print(f'Size of training labels: {y_train.shape}')
        print(
            f'Training class distribution (ground truth): {np.unique(y_train, return_counts=True)[1]/len(y_train)}'
        )
    print(
        f'Training class distribution (label model predictions): {np.unique(y_train_lm, return_counts=True)[1]/len(y_train_lm)}'
    )

    print(
        '\nKeyClass only trains on the most confidently labeled data points! Applying mask...'
    )
    print('\n==== Data statistics (after applying mask) ====')

    if training_labels_present:
        y_train_masked = y_train[mask]
    y_train_lm_masked = y_train_lm[mask]
    X_train_embed_masked = X_train_embed[mask]
    sample_weights_masked = sample_weights[mask]
    proba_preds_masked = proba_preds[mask]

    print(f'Size of training data: {X_train_embed_masked.shape}')
    if training_labels_present:
        print(f'Size of training labels: {y_train_masked.shape}')
        print(
            f'Training class distribution (ground truth): {np.unique(y_train_masked, return_counts=True)[1]/len(y_train_masked)}'
        )
    print(
        f'Training class distribution (label model predictions): {np.unique(y_train_lm_masked, return_counts=True)[1]/len(y_train_lm_masked)}'
    )

    return X_train_embed_masked, y_train_lm_masked, y_train_masked, \
     X_test_embed, y_test, \
     training_labels_present, sample_weights_masked, proba_preds_masked


def train(args_cmd):

    args = utils.Parser(config_file_path=args_cmd.config).parse()

    # Set random seeds
    random_seed = args_cmd.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X_train_embed_masked, y_train_lm_masked, y_train_masked, \
     X_test_embed, y_test, training_labels_present, \
     sample_weights_masked, proba_preds_masked = load_data(args)

    # Train a downstream classifier

    if args['use_custom_encoder']:
        encoder = models.CustomEncoder(
            pretrained_model_name_or_path=args['base_encoder'],
            device=args['device'])
    else:
        encoder = models.Encoder(model_name=args['base_encoder'],
                                 device=args['device'])

    classifier = models.FeedForwardFlexible(
        encoder_model=encoder,
        h_sizes=args['h_sizes'],
        activation=eval(args['activation']),
        device=torch.device(args['device']))
    print('\n===== Training the downstream classifier =====\n')
    model = train_classifier.train(model=classifier,
                                   device=torch.device(args['device']),
                                   X_train=X_train_embed_masked,
                                   y_train=y_train_lm_masked,
                                   sample_weights=sample_weights_masked
                                   if args['use_noise_aware_loss'] else None,
                                   epochs=args['end_model_epochs'],
                                   batch_size=args['end_model_batch_size'],
                                   criterion=eval(args['criterion']),
                                   raw_text=False,
                                   lr=eval(args['end_model_lr']),
                                   weight_decay=eval(
                                       args['end_model_weight_decay']),
                                   patience=args['end_model_patience'])

    if not os.path.exists(args['model_path']): os.makedirs(args['model_path'])
    current_time = datetime.now()
    model_name = f'end_model_{current_time.strftime("%d-%b-%Y-%H_%M_%S")}.pth'
    print(f'Saving model {model_name}...')
    with open(join(args['model_path'], model_name), 'wb') as f:
        torch.save(model, f)

    end_model_preds_train = model.predict_proba(
        torch.from_numpy(X_train_embed_masked), batch_size=512, raw_text=False)
    end_model_preds_test = model.predict_proba(torch.from_numpy(X_test_embed),
                                               batch_size=512,
                                               raw_text=False)

    # Save the predictions
    with open(join(args['preds_path'], 'end_model_preds_train.pkl'),
              'wb') as f:
        pickle.dump(end_model_preds_train, f)
    with open(join(args['preds_path'], 'end_model_preds_test.pkl'), 'wb') as f:
        pickle.dump(end_model_preds_test, f)

    # Print statistics
    if training_labels_present:
        training_metrics_with_gt = utils.compute_metrics(
            y_preds=np.argmax(end_model_preds_train, axis=1),
            y_true=y_train_masked,
            average=args['average'])
        utils.log(metrics=training_metrics_with_gt,
                  filename='end_model_with_ground_truth',
                  results_dir=args['results_path'],
                  split='train')

    training_metrics_with_lm = utils.compute_metrics(y_preds=np.argmax(
        end_model_preds_train, axis=1),
                                                     y_true=y_train_lm_masked,
                                                     average=args['average'])
    utils.log(metrics=training_metrics_with_lm,
              filename='end_model_with_label_model',
              results_dir=args['results_path'],
              split='train')

    testing_metrics = utils.compute_metrics_bootstrap(
        y_preds=np.argmax(end_model_preds_test, axis=1),
        y_true=y_test,
        average=args['average'],
        n_bootstrap=args['n_bootstrap'],
        n_jobs=args['n_jobs'])
    utils.log(metrics=testing_metrics,
              filename='end_model_with_ground_truth',
              results_dir=args['results_path'],
              split='test')

    print('\n===== Self-training the downstream classifier =====\n')

    # Fetching the raw text data for self-training
    X_train_text = utils.fetch_data(dataset=args['dataset'],
                                    path=args['data_path'],
                                    split='train')
    X_test_text = utils.fetch_data(dataset=args['dataset'],
                                   path=args['data_path'],
                                   split='test')

    model = train_classifier.self_train(
        model=model,
        X_train=X_train_text,
        X_val=X_test_text,
        y_val=y_test,
        device=torch.device(args['device']),
        lr=eval(args['self_train_lr']),
        weight_decay=eval(args['self_train_weight_decay']),
        patience=args['self_train_patience'],
        batch_size=args['self_train_batch_size'],
        q_update_interval=args['q_update_interval'],
        self_train_thresh=eval(args['self_train_thresh']),
        print_eval=True)

    current_time = datetime.now()
    model_name = f'end_model_self_trained_{current_time.strftime("%d %b %Y %H:%M:%S")}.pth'
    print(f'Saving model {model_name}...')
    with open(join(args['model_path'], model_name), 'wb') as f:
        torch.save(model, f)

    end_model_preds_test = model.predict_proba(
        X_test_text, batch_size=args['self_train_batch_size'], raw_text=True)

    # Save the predictions
    with open(
            join(args['preds_path'], 'end_model_self_trained_preds_test.pkl'),
            'wb') as f:
        pickle.dump(end_model_preds_test, f)

    # Print statistics
    testing_metrics = utils.compute_metrics_bootstrap(
        y_preds=np.argmax(end_model_preds_test, axis=1),
        y_true=y_test,
        average=args['average'],
        n_bootstrap=args['n_bootstrap'],
        n_jobs=args['n_jobs'])
    utils.log(metrics=testing_metrics,
              filename='end_model_with_ground_truth_self_trained',
              results_dir=args['results_path'],
              split='test')
    return testing_metrics


def test(args_cmd, end_model_path, end_model_self_trained_path):

    args = utils.Parser(config_file_path=args_cmd.config).parse()

    # Set random seeds
    random_seed = args_cmd.random_seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    X_train_embed_masked, y_train_lm_masked, y_train_masked, \
     X_test_embed, y_test, training_labels_present, \
     sample_weights_masked, proba_preds_masked = load_data(args)

    model = torch.load(end_model_path)

    end_model_preds_train = model.predict_proba(
        torch.from_numpy(X_train_embed_masked), batch_size=512, raw_text=False)
    end_model_preds_test = model.predict_proba(torch.from_numpy(X_test_embed),
                                               batch_size=512,
                                               raw_text=False)

    # Print statistics
    if training_labels_present:
        training_metrics_with_gt = utils.compute_metrics(
            y_preds=np.argmax(end_model_preds_train, axis=1),
            y_true=y_train_masked,
            average=args['average'])
        print('training_metrics_with_gt', training_metrics_with_gt)

    training_metrics_with_lm = utils.compute_metrics(y_preds=np.argmax(
        end_model_preds_train, axis=1),
                                                     y_true=y_train_lm_masked,
                                                     average=args['average'])
    print('training_metrics_with_lm', training_metrics_with_lm)

    testing_metrics = utils.compute_metrics_bootstrap(
        y_preds=np.argmax(end_model_preds_test, axis=1),
        y_true=y_test,
        average=args['average'],
        n_bootstrap=args['n_bootstrap'],
        n_jobs=args['n_jobs'])
    print('testing_metrics', testing_metrics)

    print('\n===== Self-training the downstream classifier =====\n')

    # Fetching the raw text data for self-training
    X_train_text = utils.fetch_data(dataset=args['dataset'],
                                    path=args['data_path'],
                                    split='train')
    X_test_text = utils.fetch_data(dataset=args['dataset'],
                                   path=args['data_path'],
                                   split='test')

    model = torch.load(end_model_self_trained_path)

    end_model_preds_test = model.predict_proba(
        X_test_text, batch_size=args['self_train_batch_size'], raw_text=True)

    # Print statistics
    testing_metrics = utils.compute_metrics_bootstrap(
        y_preds=np.argmax(end_model_preds_test, axis=1),
        y_true=y_test,
        average=args['average'],
        n_bootstrap=args['n_bootstrap'],
        n_jobs=args['n_jobs'])
    print('testing_metrics after self train', testing_metrics)
    return testing_metrics