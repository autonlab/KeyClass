from os.path import join
import re
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode(model, tokenizer, text):
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to('cuda')
        # encoded_input = tokenizer(text.tolist(), return_tensors='pt').to('cuda')
        # print(encoded_input); return None
        model_output  = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask']).cpu().numpy()
        del encoded_input
        # print(sentence_embeddings.shape)
        return sentence_embeddings

def cleantext(text):
    text = text.lower()
    text = re.sub(r'<.*?>|[\.`\',;\?\*\[\]\(\)-:_]*|[0-9]*', '', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def fetch_data(dataset='imdb', path='~/', split='train'):
    """Fetches a dataset by its name

	    Parameters
	    ---------- 
	    dataset: str
	        List of text to be encoded. 

	    path: str
	        Path to the stored data. 

	    split: str
	        Whether to fetch the train or test dataset. Options are one of 'train' or 'test'. 
    """
    _dataset_names = ['agnews', 'amazon', 'dbpedia', 'imdb', 'mimic'] 
    if dataset not in _dataset_names:
        raise ValueError(f'Dataset must be one of {_dataset_names}, but received {dataset}.')
    if split not in ['train', 'test']:
        raise ValueError(f'split must be one of \'train\' or \'test\', but received {split}.')

    text = open(f'{join(path, dataset, split)}.txt').readlines()

    if dataset == 'mimic':
        text = [cleantext(line) for line in text]

    return text 

# class Encoder:
#     def __init__(self, base_encoder='all-mpnet-base-v2', device = "cuda"):
#         """Encoder class returns an instance of a sentence transformer.
            
            
#             https://www.sbert.net/docs/pretrained_models.html
            
#             Parameters
#             ---------- 
#             base_encoder: str
#                 The pre-trained tranformer model to use for encoding text. 
#             device: str
#                 Device to use for encoding. 'cuda' by default. 
#         """
#         self.base_encoder = base_encoder
#         if self.base_encoder == 'bluebert':
#             self.tokenizer = AutoTokenizer.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
#             self.model = AutoModel.from_pretrained("bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12")
#             if device=='cuda':
#                 self.model = self.model.cuda()
#         else:
#             self.model = SentenceTransformer(model_name_or_path=base_encoder, device=device)
            
    
#     def get_embeddings(self, text, batch_size=32):
#         """Encode text.
        
#         Parameters
#         ---------- 
#         text: list
#             List of text to be encoded.
#         """
        
#         if self.base_encoder == 'bluebert':
#             embeddings = [] 
#             for i in tqdm(range(0, len(text), batch_size)):
#                 embeddings.append(encode(self.model, self.tokenizer, text[i:i+batch_size]))
#             embeddings = np.concatenate(embeddings)

#         else:
#             embeddings = self.model.encode(text, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)
        
#         return embeddings    