from importlib import reload
import data
import createLFs
import trainLabelModel
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import argparse

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
    
    data_path = '/zfsauton/project/public/chufang/classes/16811/project/data/'
    dataset = 'imdb'
    model_name = 'paraphrase-mpnet-base-v2'
    min_df = 0.001
    ngram_range = (1,3)
    topk=300
    

    train_text = data.fetch_data(dataset = dataset, path = data_path, split = 'train')
    y_train = open(data_path+dataset+'/train_labels.txt', 'r').readlines()
    y_train = np.array([int(i.replace('\n','')) for i in y_train])

    X_train = pickle.load(open(data_path+dataset+'/train_embeddings_'+model_name+'.pkl', 'rb'))

    if dataset == 'imdb':
#         label_names = ['negative hate poor ridiculous horrible boring dumb bad, would not recommend', 'positive excellent amazing interesting love fine good, would recommend', ]
        label_names = ['negative, hate, expensive, bad, poor, broke, waste, horrible, would not recommend', 'good, positive, excellent, amazing, love, fine, good quality, would recommend']
    elif dataset == 'agnews':
        label_names = ['politics', 'sports', 'business', 'technology']
    elif dataset == 'amazon':
#         label_names = ['negative, bad, hate, poor, ridiculous, horrible, bad quality', 'good, positive, excellent, amazing, love, fine, good quality', ]
        label_names = ['negative, hate, expensive, bad, poor, broke, waste, horrible, would not recommend', 'good, positive, excellent, amazing, love, fine, good quality, would recommend']
    elif dataset == 'dbpedia':
        label_names = ['company', 'school university', 'artist', 'athlete', 'politics', 
                       'transportation', 'building structure tower', 'lake river mountain',  'village town rural settlement', 'animal mammal',
                       'plant, wood, tree', 'album record music audio', 'film movie actor', 'book novel publication']
    else:
        raise NotImplementedError('Must be a valid dataset')

    print('Dataset', dataset, label_names)
    print('Number of Documents', len(train_text))
    print('y_train class distribution', np.unique(y_train, return_counts=True))

    labeler = createLFs.CreateLabellingFunctions(text_corpus=train_text, label_names=label_names, model_name=model_name, device='cuda')


    ## using just whether keyword exists -----------------------------------------
    labeler.get_label_embeddings()
    labeler.get_vocabulary(min_df=min_df, ngram_range=ngram_range)
    print('Label vocabulary:\n', len(labeler.vocabulary))
    print('labeler.word_indicator_matrix.shape', labeler.word_indicator_matrix.shape)
    # labeler.assign_categories_to_keywords(cutoff=0.9)
    keywords, assigned_category = labeler.assign_categories_to_keywords(topk=topk, label_spreading=False)
    print('Len keywords', len(keywords)) 
    print('Unique assigned_category', np.unique(assigned_category, return_counts=True))
    for u in range(len(label_names)):
        inds = np.where(assigned_category==u)[0]
        print(label_names[u], keywords[inds])
    labeler.create_label_matrix()


    # # using bert embeddings instead of just whether keyword exists -----------------------------------------
    # labeler.get_label_embeddings()
    # labeler.get_vocabulary(min_df = 0.001, ngram_range=(1,3))
    # keywords, assigned_category = labeler.assign_categories_to_keywords(topk=300)

    # print(len(keywords), len(assigned_category))
    # label_descriptions = [[l] for l in label_names]
    # for u in range(len(label_names)):
    #     inds = np.where(assigned_category==u)[0]
    #     label_descriptions[u].extend(list(keywords[inds]))
    # print('label_descriptions', [len(l) for l in label_descriptions], label_descriptions)
    # pickle.dump(label_descriptions, open(dataset+'_label_descriptions.pkl', 'wb'))
    # label_descriptions = pickle.load(open(dataset+'_label_descriptions.pkl', 'rb'))
    # labeler = createLFs.CreateLabellingFunctions(text_corpus=train_text, label_descriptions=label_descriptions)
    # labeler.get_label_embeddings()
    # labeler.assign_texts_to_keywords(text_embeddings=X_train, cutoff=97.5)



#     print('labeler.label_matrix', np.unique(labeler.label_matrix, return_counts=True))
#     label_model = trainLabelModel.LabelModelWrapper(label_matrix=labeler.label_matrix, n_classes=len(np.unique(y_train)), y_train=y_train, device='cuda')
        
#     label_model.train_label_model(lr = 0.001, n_epochs = 700, cuda=True)

#     y_proba = label_model.predict_proba()
#     weights = y_proba.values
#     labelmodel_preds = np.argmax(weights, axis=1)

# #     pickle.dump(labelmodel_preds, open(dataset+'_labelmodel_preds_'+model_name+'.pkl', 'wb'))
#     pickle.dump(weights, open('../proba_labels/'+dataset+'_weights_'+model_name+'.pkl', 'wb'))


#     print('Acc', np.mean(labelmodel_preds==y_train))
#     print('labelmodel_preds', np.unique(labelmodel_preds, return_counts=True))
