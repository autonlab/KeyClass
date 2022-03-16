## Functions to get data and prepare embeddings

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from utils import *

class Encoder:
    def __init__(self, model_name='all-mpnet-base-v2', device = "cuda"):
        """Encoder class returns an instance of a sentence transformer.
            
            
            https://www.sbert.net/docs/pretrained_models.html
            
            Parameters
            ---------- 
            model_name: str
                The pre-trained tranformer model to use for encoding text. 
            device: str
                Device to use for encoding. 'cuda' by default. 
        """
        self.model_name = model_name
        if self.model_name == 'bluebert':
            self.tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
            self.model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
            if device=='cuda':
                self.model = self.model.cuda()
        else:
            self.model = SentenceTransformer(model_name_or_path=model_name, device=device)
            
    
    def get_embeddings(self, text, batch_size=32):
        """Encode text.
        
        Parameters
        ---------- 
        text: list
            List of text to be encoded.
        path: str
            The path to save the embeddings
        name: str
            The name of the embedding pickle file
        """
        
        if self.model_name == 'bluebert':
            embeddings = [] 
            for i in tqdm(range(0, len(text), batch_size)):
                embeddings.append(encode(self.model, self.tokenizer, text[i:i+batch_size]))
            embeddings = np.concatenate(embeddings)

        else:
            embeddings = self.model.encode(text, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
        
        return embeddings