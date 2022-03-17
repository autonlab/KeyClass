import sys
sys.path.append('../keyclass/')

import utils
import create_lfs
import numpy as np
import pickle
import argparse
import torch
import os

if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--data_path', default='/zfsauton/project/public/chufang/classes/16811/project/data/',
#                         help='dataset directory')
#     parser.add_argument('--dataset', default='agnews',
#                         help='dataset')
#     parser.add_argument('--model_name', default='paraphrase-mpnet-base-v2'
#                         help='Sentence Encoder model to use')
#     parser.add_argument('--topk', default=300, type=int,
#                         help='topk keywords to use from each category')
#     args = parser.parse_args()
    
    data_path = '/zfsauton/project/public/chufang/classes/'
    dataset = 'imdb'
    model_name = 'paraphrase-mpnet-base-v2'
    min_df = 0.001
    ngram_range = (1,3)
    topk=300
    label_model_lr=0.001
    label_model_n_epochs=400
    save_results_path = '../results/'+dataset+'/'
    # end of params ---------------------------------------------------------------
    
    # load training data
    train_text = utils.fetch_data(dataset=dataset, path=data_path, split='train')

    y_train = open(data_path+dataset+'/train_labels.txt', 'r').readlines()
    y_train = np.array([int(i.replace('\n','')) for i in y_train])

    X_train = pickle.load(open(data_path+dataset+'/train_embeddings.pkl', 'rb'))

    if dataset == 'imdb':
        label_names = ['negative, hate, expensive, bad, poor, broke, waste, horrible, would not recommend', 
            'good, positive, excellent, amazing, love, fine, good quality, would recommend']

    elif dataset == 'agnews':
        label_names = ['politics', 'sports', 'business', 'technology']

    elif dataset == 'amazon':
        label_names = ['negative, hate, expensive, bad, poor, broke, waste, horrible, would not recommend', 
            'good, positive, excellent, amazing, love, fine, good quality, would recommend']

    elif dataset == 'dbpedia':
        label_names = ['company', 'school university', 'artist', 'athlete', 'politics',
            'transportation', 'building structure tower', 'lake river mountain',  'village town rural settlement', 'animal mammal',
            'plant, wood, tree', 'album record music audio', 'film movie actor', 'book novel publication']

    else:
        raise NotImplementedError('Must be a valid dataset')

    print('dataset, label_names', dataset, label_names)
    print('len(train_text)', len(train_text))
    print('y_train class distribution', np.unique(y_train, return_counts=True))


    # creating labeling functions
    labeler = create_lfs.CreateLabellingFunctions(model_name=model_name)
    proba_preds = labeler.get_labels(text_corpus=train_text, label_names=label_names, min_df=min_df, 
        ngram_range=ngram_range, topk=topk, y_train=y_train, 
        label_model_lr=label_model_lr, label_model_n_epochs=label_model_n_epochs, verbose=True)

    y_train_pred = np.argmax(proba_preds, axis=1)

    print('Label Model Training Accuracy', np.mean(y_train_pred==y_train))
    print('Label Model Predictions: Unique and Counts', np.unique(y_train_pred, return_counts=True))

    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    pickle.dump(proba_preds, open(save_results_path+'proba_preds.pkl', 'wb'))
