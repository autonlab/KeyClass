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

from curses import raw
import torch
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
from tqdm import tqdm, trange
import copy


def get_q_soft(p: np.ndarray):
    """Get target distribution for model refinement via self-training. 

    Soft labeling (Xie et al., 2016) derives Q by enhancing high-confidence predictions while
    demoting low-confidence ones via squaring and normalizing the current predictions.

    Parameters
    ----------
    p: Current predictions of the model.

    References
    ----------
    Junyuan Xie, Ross B. Girshick, and Ali Farhadi. 2016. Unsupervised deep embedding for clustering analysis. In ICML.
    """
    q = np.square(p) / np.sum(p, axis=0, keepdims=True)
    q = q / np.sum(q, axis=1, keepdims=True)
    return q


def train(model: torch.nn.Module,
          X_train: Union[Union[str, List[str]], np.ndarray],
          y_train: Union[torch.Tensor, np.ndarray],
          device: torch.device = torch.device("cuda"),
          sample_weights: Optional[np.array] = None,
          epochs: int = 200,
          batch_size: int = 128,
          criterion: Callable = torch.nn.CrossEntropyLoss(reduction='none'),
          raw_text: bool = False,
          lr: float = 1e-3,
          weight_decay: float = 1e-4,
          patience: int = 2):
    """Function to train the encoder along with fully connected layers. 

    Parameters
    ----------
    model: ML/DL Model to train
    X_train: Training Data Features
    y_train: Training Data Ground Truth
    device: Device to use for training. 'cuda' by default
    sample_weights: Array of weights assigned to individual samples
    epochs: Number of complete passes of the training data through the model
    batch_size: Number of samples to feed into the model before updating hyperparameters
    criterion: Loss function (or Optimizer)
    raw_text: Boolean Flag describing if raw text is to be processed (True if processing raw text, else False)
    lr: Learning Rate
    weight_decay: Weight decay parameter (for regularization/to prevent overfitting) 
    patience: Number of consecutive epochs of no performance improvement before terminating training (for early stopping)
    """
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train)

    if raw_text == False and isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train)

    if sample_weights is not None:
        sample_weights = torch.from_numpy(sample_weights.reshape(
            -1, 1)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model = model.train()

    best_loss = np.inf
    tolcount = 0
    best_state_dict = None

    N = len(X_train)
    pbar = trange(epochs, unit="batch")
    for nep in pbar:
        pbar.set_description(f"Epoch {nep}")
        permutation = torch.randperm(N)
        running_loss = 0

        for i in range(0, N, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i + batch_size]

            batch_x, batch_y = X_train[indices], y_train[indices]
            batch_y = batch_y.to(device)
            # Since raw text is a list of strings, it cannot be trivially moved to the GPU using the
            # .to() method. The base encoder model takes care of this.
            if raw_text == False: batch_x = batch_x.to(device)

            out = model.forward(batch_x, mode='inference', raw_text=raw_text)
            loss = criterion(out, batch_y)

            if sample_weights is not None:
                batch_weight = sample_weights[indices]
                loss = torch.mul(loss, batch_weight).mean()
            else:
                loss = loss.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss = running_loss + (loss.cpu().detach().numpy() *
                                           batch_size / N)

        scheduler.step()

        with torch.no_grad():  # Early stopping
            pbar.set_postfix(tolerance_count=tolcount,
                             running_loss=running_loss,
                             best_loss=best_loss)
            if running_loss <= best_loss:
                best_loss = running_loss
                tolcount = 0
                best_state_dict = copy.deepcopy(model.state_dict())

            else:  # loss.cpu().detach().numpy() > best_loss:
                tolcount += 1

            if tolcount > patience:
                print('Stopping early...')
                model.load_state_dict(best_state_dict)  # Return the best model
                return model

    return model


def self_train(model: torch.nn.Module,
               X_train: Union[str, List[str]],
               X_val: Union[str, List[str]],
               y_val: np.ndarray,
               device: torch.device = torch.device("cuda"),
               lr: float = 1e-5,
               weight_decay: float = 1e-4,
               batch_size: int = 32,
               q_update_interval: int = 50,
               patience: int = 3,
               self_train_thresh: float = 1 - 2e-3,
               print_eval: bool = True):
    """Function to self train a model.

    Parameters
    ----------
    model: ML/DL model to self train on
    X_train: Feature vectors for training dataset
    X_val: Feature vectors for validation
    y_val: Ground Truths for validation
    device: Device to use for self training. 'cuda' by default
    lr: Learning Rate for self training
    weight_decay: Weight decay parameter (for regularization/to prevent overfitting) for self training
    batch_size: Number of samples to feed into the model before updating hyperparameters for self training
    q_update_interval: Number of steps before q is updated for self training
    patience: Number of consecutive epochs of no performance improvement before terminating training (for early stopping) for self training
    self_train_thresh: If p matches q at a rate above this threshold for "patience" number of epochs, then self training will stop early (if predictions p are not flipping, stop early)
    print_eval: Boolean - prints validation metrics if True, and does not if False
    """
    model.train()
    model.zero_grad()
    model.to(device)

    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    tolcount = 0

    # Update P every batch and Q every epoch
    N = len(X_train)
    permutation = torch.randperm(N)

    X_train = np.array(
        X_train)  # Ensures that we are able to index into X_train

    pbar = trange(N // (batch_size * q_update_interval), unit="batch")
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        inds = np.random.randint(0, N, batch_size * q_update_interval)

        with torch.no_grad():
            pred_proba = model.predict_proba(X_train[inds],
                                             batch_size=batch_size,
                                             raw_text=True)
            target_dist = get_q_soft(
                pred_proba)  # should be of size (N, num_categories)
            target_preds = np.argmax(target_dist, axis=1)

            self_train_agreement = np.mean(
                np.argmax(pred_proba, axis=1) == target_preds)

            if self_train_agreement > self_train_thresh: tolcount += 1
            else: tolcount = 0

            if tolcount >= patience:
                break

        for i in range(0, batch_size * q_update_interval, batch_size):
            batch_x = X_train[inds][
                i:i +
                batch_size]  # The training data is moved to device by the encoder model in its forward function
            batch_q = torch.from_numpy(target_dist[i:i +
                                                   batch_size]).to(device)

            out = model.forward(batch_x, mode='self_train', raw_text=True)
            loss = criterion(out, batch_q)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            del batch_x, batch_q

        if print_eval == True:
            val_preds = model.predict(X_val)
            # print('tolcount', tolcount, 'self_train_agreement', self_train_agreement, 'validation_accuracy', np.mean(val_preds==y_val))

        pbar.set_postfix(tolerance_count=tolcount,
                         self_train_agreement=self_train_agreement,
                         validation_accuracy=np.mean(
                             val_preds == y_val) if print_eval else None)
    return model
