import sys
sys.path.append('../keyclass/')

import numpy as np
import torch
import os
from os.path import join
import models
import utils
import train_classifier
from config import Parser
import pickle as pkl
from datetime import datetime

parser = Parser(config_file_path='../default_config.yml')
args = parser.parse()

with open(join(args['preds_path'], 'proba_preds.pkl'), 'rb') as f:
    proba_preds = pkl.load(f)
y_train_lm = np.argmax(proba_preds, axis=1)
sample_weights = np.max(proba_preds, axis=1) # Sample weights for noise aware loss

# Keep only very confident predictions
mask = utils.get_balanced_data_mask(proba_preds, max_num=args['max_num'], class_balance=None)

# Load training and testing data 
# We have already encode the dataset, so we'll just load the embeddings
with open(join(args['data_path'], args['dataset'], f'train_embeddings.pkl'), 'rb') as f:
        X_train_embed = pkl.load(f)
with open(join(args['data_path'], args['dataset'], f'test_embeddings.pkl'), 'rb') as f:
        X_test_embed = pkl.load(f)

# Load training and testing ground truth labels
with open(join(args['data_path'], args['dataset'], f'train_labels.txt'), 'r') as f:
        y_train = f.readlines()
y_train = np.array([int(i.replace('\n','')) for i in y_train])
with open(join(args['data_path'], args['dataset'], f'test_labels.txt'), 'r') as f:
        y_test = f.readlines()
y_test = np.array([int(i.replace('\n','')) for i in y_test])

# Print data statistics
print('\n==== Data statistics ====')
print(f'Size of training data: {X_train_embed.shape}, testing data: {X_test_embed.shape}')
print(f'Size of training labels: {y_train.shape}, testing labels: {y_test.shape}')
print(f'Training class distribution (ground truth): {np.unique(y_train, return_counts=True)[1]/len(y_train)}')
print(f'Training class distribution (label model predictions): {np.unique(y_train_lm, return_counts=True)[1]/len(y_train_lm)}')

print('\nKeyClass only trains on the most confidently labeled data points! Applying mask...')
print('\n==== Data statistics (after applying mask) ====')

y_train_masked = y_train[mask]
y_train_lm_masked = y_train_lm[mask]
proba_preds_masked = proba_preds[mask]
X_train_embed_masked = X_train_embed[mask]
sample_weights_masked = sample_weights[mask]

print(f'Size of training data: {X_train_embed_masked.shape}')
print(f'Size of training labels: {y_train_masked.shape}')
print(f'Training class distribution (ground truth): {np.unique(y_train_masked, return_counts=True)[1]/len(y_train_masked)}')
print(f'Training class distribution (label model predictions): {np.unique(y_train_lm_masked, return_counts=True)[1]/len(y_train_lm_masked)}')

# Train a downstream classifier

encoder = models.Encoder(model_name=args['base_encoder'], 
						 device=args['device'])

classifier = models.FeedForwardFlexible(encoder_model=encoder,
										h_sizes=args['h_sizes'], 
										activation=eval(args['activation']),
										device=torch.device(args['device']))

model = train_classifier.train(model=classifier, 
							   device=torch.device(args['device']),
							   X_train=X_train_embed_masked, 
							   y_train=y_train_lm_masked,
							   sample_weights=sample_weights_masked if args['use_noise_aware_loss'] else None, 
					           epochs=args['epochs'], 
					           batch_size=args['batch_size'], 
					           criterion=eval(args['criterion']), 
					           raw_text=False, 
					           lr=eval(args['lr']), 
					           weight_decay=eval(args['weight_decay']),
					           patience=args['patience'])

if not os.path.exists(args['model_path']): os.makedirs(args['model_path'])
current_time = datetime.now()
model_name = f'end_model_{current_time.strftime("%d-%b-%Y-%H_%M_%S")}.pth'
print(f'Saving model {model_name}...')
with open(join(args['model_path'], model_name), 'wb') as f:
    torch.save(model, f)

end_model_preds_train = model.predict_proba(torch.from_numpy(X_train_embed_masked), batch_size=512, raw_text=False)
end_model_preds_test = model.predict_proba(torch.from_numpy(X_test_embed), batch_size=512, raw_text=False)

# Save the predictions
with open(join(args['preds_path'], 'end_model_preds_train.pkl'), 'wb') as f:
    pkl.dump(end_model_preds_train, f)

with open(join(args['preds_path'], 'end_model_preds_test.pkl'), 'wb') as f:
    pkl.dump(end_model_preds_test, f)

# Print statistics
training_metrics_with_gt = utils.compute_metrics(y_preds=np.argmax(end_model_preds_train, axis=1), 
	y_true=y_train_masked, average=args['average'])
utils.log(metrics=training_metrics_with_gt, 
	filename='end_model_with_ground_truth', 
	results_dir=args['results_path'], split='train')

training_metrics_with_lm = utils.compute_metrics(y_preds=np.argmax(end_model_preds_train, axis=1), 
	y_true=y_train_lm_masked, average=args['average'])
utils.log(metrics=training_metrics_with_lm, 
	filename='end_model_with_label_model', 
	results_dir=args['results_path'], split='train')

testing_metrics = utils.compute_metrics_bootstrap(y_preds=np.argmax(end_model_preds_test, axis=1), 
	y_true=y_test, average=args['average'], n_bootstrap=args['n_bootstrap'], n_jobs=args['n_jobs'])
utils.log(metrics=testing_metrics, 
	filename='end_model_with_ground_truth', 
	results_dir=args['results_path'], split='test')
