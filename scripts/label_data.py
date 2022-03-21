import sys
sys.path.append('../keyclass/')

import utils
import create_lfs
import numpy as np
import pickle
import argparse
import torch
from os.path import join
from config import Parser

parser = Parser()
args = parser.parse()

# load training data
train_text = utils.fetch_data(dataset=args.dataset, path=args.data_path, split='train')

with open(join(args.data_path, args.dataset, 'train_labels.txt'), 'rb') as f:
    y_train = f.readlines()
y_train = np.array([int(i.replace('\n','')) for i in y_train])

with open(join(args.data_path, args.dataset, 'train_embeddings.txt'), 'rb') as f:
    X_train = pickle.load(f)

# Print dataset statistics
print(f'Getting labels for the {args.dataset} data...')
print(f'Size of the data: {len(train_text)}')
print('Class distribution', np.unique(y_train, return_counts=True))

# creating labeling functions
labeler = create_lfs.CreateLabellingFunctions(model_name=model_name)
proba_preds = labeler.get_labels(text_corpus=train_text, label_names=label_names, min_df=min_df, 
    ngram_range=ngram_range, topk=topk, y_train=y_train, 
    label_model_lr=label_model_lr, label_model_n_epochs=label_model_n_epochs, verbose=True)

y_train_pred = np.argmax(proba_preds, axis=1...)

print('Label Model Training Accuracy', np.mean(y_train_pred==y_train))
print('Label Model Predictions: Unique and Counts', np.unique(y_train_pred, return_counts=True)
if not os.path.exists(save_results_path):   os.makedirs(save_results_path)
pickle.dump(proba_preds, open(save_results_path+'proba_preds.pkl', 'wb'))
