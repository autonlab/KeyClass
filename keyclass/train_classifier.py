from curses import raw
import torch
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
from tqdm import tqdm
import copy

def get_q_soft(p: torch.tensor):
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
    q = torch.square(p) / torch.sum(p, dim=0, keepdim=True)
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q


def train(model: torch.nn.Module, 
          device: torch.device = torch.device("cuda"),
          X_train: Union[Union[str, List[str]], np.ndarray], 
          y_train: Union[torch.Tensor, np.ndarray], 
          sample_weights: Optional[np.array],
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

    """
    if isinstance(y_train, np.ndarray):
        y_train = torch.from_numpy(y_train)

    if raw_text == False and isinstance(X_train, np.ndarray):
        X_train = torch.from_numpy(X_train)

    if sample_weights is not None:
        sample_weights = torch.from_numpy(sample_weights.reshape(-1, 1)).to(device).float()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model = model.train()

    best_loss = np.inf
    tolcount = 0
    best_state_dict = None

    N = len(X_train)
    for nep in tqdm(range(epochs)):
        permutation = torch.randperm(N)
        running_loss = 0
        
        for i in range(0, N, batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
                
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
            running_loss = running_loss + (loss.cpu().detach().numpy() * batch_size / N)

        scheduler.step()
            
        with torch.no_grad(): # Early stopping
            print('Tolerance count:', tolcount, 'Running loss:', running_loss, 'Best loss:', best_loss)
            if running_loss <= best_loss:
                best_loss = running_loss
                tolcount = 0
                best_state_dict = copy.deepcopy(model.state_dict())

            else: # loss.cpu().detach().numpy() > best_loss:
                tolcount += 1

            if tolcount > patience:
                print('Stopping early...')
                model.load_state_dict(best_state_dict) # Return the best model
                
    return model


def self_train(model, X_train, X_val, y_val, device, lr=1e-5, weight_decay=1e-4, batch_size=32, q_update_interval=50,
    patience=3, self_train_thresh=1-2e-3, print_eval=True):

    model.train()
    model.zero_grad()

    model.to(device)
    criterion = torch.nn.KLDivLoss(reduction='batchmean')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print('X_train', len(X_train))
    if type(X_train) != np.ndarray:
        X_train = np.array(X_train)
    
    tolcount = 0
    
    # update p every batch, update q every epoch
    N = len(X_train)
    permutation = torch.randperm(N)
        
    for epoch in range(N // (batch_size*q_update_interval)):
        inds = np.random.randint(0, N, batch_size*q_update_interval)
        with torch.no_grad():
            all_preds = model.predict_proba(X_train[inds], batch_size=batch_size)
            
            model.train()
            target_dist  = get_q_soft(torch.from_numpy(all_preds)) # should be of size (N, num_categories)
            target_preds = torch.argmax(target_dist, dim=1).detach().cpu().numpy()
            
            if np.mean(np.argmax(all_preds, axis=1) == target_preds) > self_train_thresh:
                tolcount += 1
            else:
                tolcount = 0

            print("Self Train Agreement:", np.mean(np.argmax(all_preds, axis=1) == target_preds), 
                    'tolcount', tolcount)

            if tolcount >= patience:
                break

        for i in range(0, batch_size*q_update_interval, batch_size):
            batch_x = X_train[inds][i:i+batch_size]
            batch_q = target_dist[i:i+batch_size].to(device)
            
            out = model.forward(batch_x, mode='self_train')
            loss = criterion(out, batch_q)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            del batch_x, batch_q
        
        if print_eval==True:
            val_preds = model.predict(X_val)
            print('Validation accuracy', np.mean(val_preds==y_val), 'tolcount', tolcount)

    return model