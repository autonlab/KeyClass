from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import sentence_transformers.util
import numpy as np
from tqdm import tqdm
import utils
import torch

class Encoder(torch.nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2', device="cuda"):
        """Encoder class returns an instance of a sentence transformer.
            https://www.sbert.net/docs/pretrained_models.html
            
            Parameters
            ---------- 
            model_name: str
                The pre-trained tranformer model to use for encoding text. 
            device: str
                Device to use for encoding. 'cuda' by default. 
        """
        super(Encoder, self).__init__()

        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.device = device

        self.to(device)

    def get_embeddings(self, text, batch_size=32):
        """
        Computes sentence embeddings, follows the logic from sentence transformers
        :param text: the text to embed
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.

        """
        embeddings = [] 
        with torch.no_grad():
            for i in tqdm(range(0, len(text), batch_size)):
                embeddings.append(self.forward(text[i:i+batch_size]).cpu().numpy())
        embeddings = np.concatenate(embeddings)

        return embeddings

    def forward(self, sentences):
        # cannot use encode due to torch no_grad in sentence transformers
#         x = self.model.encode(x, convert_to_numpy=False, convert_to_tensor=True, batch_size=len(x), show_progress_bar=False)
        all_embeddings = []

        length_sorted_idx = np.argsort([-self.model._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        features = self.model.tokenize(sentences_sorted)
        features = sentence_transformers.util.batch_to_device(features, self.device)

        out_features = self.model.forward(features)
        embeddings = out_features['sentence_embedding']
        all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = torch.stack(all_embeddings)

        return all_embeddings



class FeedforwardFlexibleRawText(torch.nn.Module):
    def __init__(self, h_sizes, activation, 
                 encoder_model, 
                 device="cuda"):
        super(FeedforwardFlexibleRawText, self).__init__()

        self.encoder_model = encoder_model
        self.device = device
        self.layers = torch.nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(p=0.5))
    
    def forward(self, x, mode='inference'):
        x = self.encoder_model(x)
        
        for layer in self.layers:
            x = layer(x)
            
        if mode=='inference':
            x = torch.nn.Softmax(dim=-1)(x)
        elif mode=='self_train':
            x = torch.nn.LogSoftmax(dim=-1)(x)
        else:
            raise AttributeError('invalid mode for forward function')

        return x

    def predict(self, Xtest, batch_size=128):
        preds = self.predict_proba(Xtest, batch_size=128)
        preds = np.argmax(preds, axis=1)
        return preds
        
    def predict_proba(self, Xtest, batch_size=128):
        with torch.no_grad():
            self.model.eval()
            probs_list = []
            self.model.to(self.device)            
            N = len(Xtest)
            
            for i in tqdm(range(0, N, batch_size)):
                probs = self.model.forward(Xtest[i:i+batch_size], mode='inference').cpu().numpy()
                probs_list.append(probs)
        return np.concatenate(probs_list, axis=0)

   
    
class TorchMLP:
    def __init__(self, h_sizes=[150, 10, 10], activation=torch.nn.ReLU(), optimizer='Adam',
                 optimparams={}, nepochs=200, device=torch.device("cpu"),
                 encoder_model_name='paraphrase-mpnet-base-v2'):
        self.encoder_model_name = encoder_model_name

        self.model = FeedforwardFlexibleRawText(h_sizes, activation,
                                                encoder_model_name='paraphrase-mpnet-base-v2', 
                                                device="cpu" if device==torch.device("cpu") else "cuda").float()
        self.optimizer = optimizer
        self.device = device

        if optimparams: 
            self.optimparams = optimparams
        else: 
            self.optimparams = {'lr': 1e-3, 'weight_decay': 1e-4}

        
    def fit(self, X, Y, batch_size, epochs, sample_weights=None, patience=2, raw_text=False):
        self.model.train()
        self.model.zero_grad()

        if raw_text == False:
            trainX = torch.from_numpy(X)
        else:
            trainX = X
        if self.encoder_model_name == 'bluebert':
            trainy = torch.from_numpy(Y).float()
        else:
            trainy = torch.from_numpy(Y.reshape(-1))

        if sample_weights is not None:
            trainweight = torch.from_numpy(sample_weights.reshape(-1, 1)).to(self.device).float()
        else:
            trainweight = None

        if self.encoder_model_name=='bluebert':
            criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
#         criterion = torch.nn.KLDivLoss(reduction='batchmean')
        
        self.model.to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), **self.optimparams)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        self.model = self.model.train()

        best_loss = np.inf
        tolcount = 0
        best_state_dict = None

        N = len(trainX)
        for nep in tqdm(range(epochs)):
            permutation = torch.randperm(N)
            running_loss = 0
            
            for i in range(0, N, batch_size):
                optimizer.zero_grad()
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = trainX[indices], trainy[indices].to(self.device)
                
                if raw_text==False:
                    batch_x = batch_x.to(self.device)
                
                batch_y = batch_y.to(self.device)
                
                out = self.model(batch_x, mode='inference', raw_text=raw_text)
#                 print(out.dtype, trainweight[indices].dtype, out.shape, trainweight[indices].shape)
#                 loss = criterion(out, trainweight[indices])
#                 print(out.dtype, batch_y.dtype)
                loss = criterion(out, batch_y)
                if trainweight is not None:
                    batch_weight = trainweight[indices]
                    loss = torch.mul(loss, batch_weight).mean()
                else:
                    loss = loss.mean()
                           
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.cpu().detach().numpy() * batch_size / N
            scheduler.step()
                
            with torch.no_grad(): # # early stopping
                print('tolcount', tolcount, 'running_loss', running_loss, 'best_loss', best_loss)
                if running_loss <= best_loss:
                    best_loss = running_loss
                    tolcount = 0
                    best_state_dict = copy.deepcopy(self.model.state_dict())

                    torch.save(self.model, 'best_model.pt')
                else: # loss.cpu().detach().numpy() > best_loss:
                    tolcount += 1

                if tolcount > patience:
                    print('Stopping early')
                    self.model.load_state_dict(best_state_dict)

                    break
                                
                
    def get_q_soft(self, p):
        q = torch.square(p) / torch.sum(p, dim=0, keepdim=True)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    

    
    def self_train(self, X_train, X_val, y_val, lr=1e-5, batch_size=32, q_update_interval=50, 
                   patience=3, self_train_thresh=1-2e-3, print_eval=True):
        self.model.train()
        self.model.zero_grad()

        self.optimparams['lr'] = lr
        self.model.to(self.device)
        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        
        optimizer = torch.optim.Adam(self.model.parameters(), **self.optimparams)
#         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
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
                all_preds = self.predict_proba(X_train[inds], batch_size=batch_size)
                self.model.train()
                target_dist  = self.get_q_soft(torch.from_numpy(all_preds)) # should be of size (N, num_categories)
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
                batch_q = target_dist[i:i+batch_size].to(self.device)
                
                out = self.model(batch_x, mode='self_train')
                loss = criterion(out, batch_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                del batch_x, batch_q
            
            if print_eval==True:
                val_preds = self.predict(X_val)
                print('Validation accuracy', np.mean(val_preds==y_val), 'tolcount', tolcount)

        return self.model
    
    
    def predict(self, Xtest, batch_size=128):
        preds = self.predict_proba(Xtest, batch_size=128)
        preds = np.argmax(preds, axis=1)
        return preds
        
    def predict_proba(self, Xtest, batch_size=128):
        self.model.eval()
        with torch.no_grad():
            out_list = []
            self.model.to(self.device)            
            N = len(Xtest)
            
            for raw_i, i in enumerate(tqdm(range(0, N, batch_size))):        
                batch_x = Xtest[i:i+batch_size]    
                out = self.model(batch_x, mode='inference')
                out_list.append(out.detach().cpu().numpy())
        return np.concatenate(out_list, axis=0)

