from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import sentence_transformers.util
import numpy as np
from tqdm import tqdm
import torch

class Encoder(torch.nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2', device="cpu"):
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
            self.model.eval()
            probs_list = []
            self.model.to(self.device)            
            N = len(Xtest)
            
            for i in tqdm(range(0, N, batch_size)):
                probs = self.model.forward(Xtest[i:i+batch_size], mode='inference').cpu().numpy()
                probs_list.append(probs)
        return np.concatenate(probs_list, axis=0)
