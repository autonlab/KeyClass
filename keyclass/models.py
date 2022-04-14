from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import sentence_transformers.util
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import numpy as np
from tqdm import tqdm, trange
from tqdm.autonotebook import trange
import torch
import logging 

logger = logging.getLogger(__name__)

class Encoder(torch.nn.Module):
    def __init__(self, model_name: str ='all-mpnet-base-v2', 
                 device: str = "cuda"):
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

    def encode(self, sentences: Union[str, List[str]], 
               batch_size: int = 32, 
               show_progress_bar: Optional[bool] = None, 
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

    def forward(self, sentences: Union[str, List[str]], 
                batch_size: int = 32, 
                show_progress_bar: Optional[bool] = None, 
                normalize_embeddings: bool = False):
        """
        Computes sentence embeddings

        
        Parameters
        ---------- 
        sentences: the sentences to embed
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

class FeedForwardFlexible(torch.nn.Module):
    def __init__(self, encoder_model: torch.nn.Module,
                 h_sizes: Iterable[int] = [768, 256, 64, 2], 
                 activation: torch.nn.Module = torch.nn.LeakyReLU(), 
                 device: str = "cuda"):
        super(FeedForwardFlexible, self).__init__()
        """
        Flexible feed forward network over a base encoder. 

        
        Parameters
        ---------- 
        encoder_model: The base encoder model
        h_sizes: Linear layer sizes to be used in the MLP
        activation: Activation function to be use in the MLP. 
        device: Device to use for training. 'cpu' by default.
        """

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
        else:
            x.to(self.device)
        
        for layer in self.layers:
            x = layer(x)
            
        if mode == 'inference':
            x = torch.nn.Softmax(dim=-1)(x)
        elif mode == 'self_train':
            x = torch.nn.LogSoftmax(dim=-1)(x)
        
        return x

    def predict(self, x_test, batch_size=128, raw_text=True):
        preds = self.predict_proba(x_test, batch_size=batch_size, raw_text=raw_text)
        preds = np.argmax(preds, axis=1)
        return preds
        
    def predict_proba(self, x_test, batch_size=128, raw_text=True):
        with torch.no_grad():
            self.eval()
            probs_list = []
            N = len(x_test)
            for i in trange(0, N, batch_size, unit='batches'):
                test_batch = x_test[i:i+batch_size] 
                probs = self.forward(test_batch, mode='inference', raw_text=raw_text).cpu().numpy()
                probs_list.append(probs)
            self.train()
        return np.concatenate(probs_list, axis=0)
