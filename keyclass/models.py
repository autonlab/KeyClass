from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import sentence_transformers.util
import numpy as np
from tqdm import tqdm
from tqdm.autonotebook import trange
import torch
import logging 

logger = logging.getLogger(__name__)

class Encoder(torch.nn.Module):
    def __init__(self, model_name: str ='all-mpnet-base-v2', device: str = "cpu"):
        """Encoder class returns an instance of a sentence transformer.
            https://www.sbert.net/docs/pretrained_models.html
            
            Parameters
            ---------- 
            model_name: str
                The pre-trained tranformer model to use for encoding text. 
            device: str
                Device to use for encoding. 'cpu' by default. 
        """
        super(Encoder, self).__init__()

        self.model_name = model_name
        self.model = SentenceTransformer(model_name_or_path=model_name, device=device)
        self.device = device

        self.to(device)

    def encode(self, sentences, batch_size: int = 32, 
               show_progress_bar: bool = None, 
               normalize_embeddings: bool = False):
        """
        Computes sentence embeddings using the forward function

        Parameters
        ---------- 
        text: the text to embed
        batch_size: the batch size used for the computation
        """
        self.model.eval() # Set model in evaluation mode. 
        with torch.no_grad():
            embeddings = self.forward(sentences, batch_size=batch_size, 
                                      show_progress_bar=show_progress_bar, 
                                      normalize_embeddings=normalize_embeddings).detach().cpu().numpy()
        self.model.train()
        return embeddings

    def forward(self, sentences, batch_size: int = 32, 
                show_progress_bar: bool = None, 
                normalize_embeddings: bool = False):
        """
        Computes sentence embeddings

        
        Parameters
        ---------- 
        text: the text to embed
        batch_size: the batch size used for the computation
        show_progress_bar: Output a progress bar when encode sentences
        normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        """
        # Cannot use encode due to torch no_grad in sentence transformers
        # x = self.model.encode(x, convert_to_numpy=False, convert_to_tensor=True, batch_size=len(x), show_progress_bar=False)
        # Logic from https://github.com/UKPLab/sentence-transformers/blob/8822bc4753849f816575ab95261f5c6ab7c71d01/sentence_transformers/SentenceTransformer.py#L110
        
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        all_embeddings = []

        length_sorted_idx = np.argsort([-self.model._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.model.tokenize(sentences_batch)
            features = sentence_transformers.util.batch_to_device(features, self.device)

            out_features = self.model.forward(features)
            
            embeddings = out_features['sentence_embedding']
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        all_embeddings = torch.stack(all_embeddings) # Converts to tensor

        return all_embeddings

class TorchMLP(torch.nn.Module):
    def __init__(self, h_sizes, activation, 
                 encoder_model, 
                 device="cpu"):
        super(TorchMLP, self).__init__()

        self.encoder_model = encoder_model
        self.device = device
        self.layers = torch.nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(activation)
            self.layers.append(torch.nn.Dropout(p=0.5))

        self.to(device)
    
    def forward(self, x, mode='inference', raw_text=True):
        if raw_text:
            x = self.encoder_model.forward(x)
        
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
        preds = self.predict_proba(Xtest, batch_size=batch_size)
        preds = np.argmax(preds, axis=1)
        return preds
        
    def predict_proba(self, Xtest, batch_size=128):
        with torch.no_grad():

            # if isinstance(Xtest, np.ndarray):
            #     Xtest = torch.from_numpy(Xtest)

            self.eval()
            probs_list = []
            N = len(Xtest)
            
            for i in tqdm(range(0, N, batch_size)):
                probs = self.forward(Xtest[i:i+batch_size], mode='inference').cpu().numpy()
                probs_list.append(probs)
        return np.concatenate(probs_list, axis=0)
