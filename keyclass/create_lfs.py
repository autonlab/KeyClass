from nltk import download, pos_tag, corpus
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.semi_supervised import LabelPropagation

from encoder import Encoder


class CreateLabellingFunctions:
    """Class to create and store labelling functions.     

        Parameters
        ---------- 
        text_corpus: list
            List of documents

        label_names: list
            List of label names of classes. For e.g. ["positive", "negative"], 
            ["good", "bad"] etc. for the IMDb dataset.  

        label_descriptions: list
            Descriptions of each class label.
    """

    def __init__(self, text_corpus: list, label_names: list=[], label_descriptions: list=[], 
                    device='cpu', model_name='all-mpnet-base-v2'):
        
        self.label_names = label_names
        self.label_descriptions = label_descriptions
        self.text_corpus = text_corpus
        self.labelling_functions = [] # List of all labeling functions
        self.encoder = Encoder(model_name=model_name, device='cuda' if torch.cuda.is_available() else 'cpu')

    def get_label_embeddings(self):
        """Computes label embeddings
        """
        if len(self.label_descriptions) == 0:
            self.label_embeddings = self.encoder.get_embeddings(text=self.label_names)
        else:
            self.label_embeddings = [self.encoder.get_embeddings(text=des) for des in self.label_descriptions]

    def assign_texts_to_keywords(self, text_embeddings, cutoff=.9):
        if len(self.label_descriptions) == 0:
            print('Using label_names')
            distances = 1 - distance.cdist(text_embeddings, self.label_embeddings, 'cosine')
            distances = (distances - np.min(distances)) / (np.max(distances) - np.min(distances))
        
            self.label_matrix = np.zeros((len(self.text_corpus), len(self.label_embeddings))) - 1
            
            for u in range(len(self.label_embeddings)):
                self.label_matrix[np.where(distances[:, u] > cutoff)[0], u] = u
            self.label_matrix = pd.DataFrame(self.label_matrix, columns=self.label_names)
        
        else: 
            print('Using label_descriptions')
            all_lens = np.sum([len(l) for l in self.label_embeddings])
            all_labels = []
            for l in self.label_descriptions:
                all_labels.extend(l)
            self.label_matrix = []
            
            for u in range(len(self.label_embeddings)):
                distances = 1-distance.cdist(text_embeddings, self.label_embeddings[u], 'cosine')
                distances = np.where(distances>np.percentile(distances, q=cutoff, axis=0, keepdims=True), u, -1)
                self.label_matrix.append(distances)
            self.label_matrix = np.concatenate(self.label_matrix, axis=1)
            print('label_matrix.shape', self.label_matrix.shape)
            self.label_matrix = pd.DataFrame(self.label_matrix, columns=all_labels)

        
    def get_vocabulary(self, max_df: float = 1.0, min_df: float = 0.01, 
        ngram_range: tuple = (1, 1), return_vocabulary: bool = False):
        """Returns vocabulary and word indicator matrix
        The word indicator matrix is a n x m matrix corresponding to n documents and m words in the
        vocabulary. 

        """
        # Vectorizing the vocabulary
        vectorizer = CountVectorizer(max_df = max_df, min_df = min_df, strip_accents = 'unicode',
            stop_words = corpus.stopwords.words('english'), ngram_range = ngram_range)
        word_indicator_matrix = vectorizer.fit_transform(self.text_corpus)
        self.word_indicator_matrix = word_indicator_matrix.toarray()

        self.vocabulary = np.asarray(vectorizer.get_feature_names()) # Vocabulary
        self.keywords = self.vocabulary

        if return_vocabulary:
            return self.vocabulary, self.word_indicator_matrix

    def assign_categories_to_keywords(self, cutoff=None, topk=None, min_topk=True, label_spreading=False):
        assert((cutoff is None) or (topk is None))
        if not label_spreading:
            self.vocabulary_embeddings = self.encoder.get_embeddings(text=self.vocabulary, save=False)
            distances = distance.cdist(self.vocabulary_embeddings, self.label_embeddings, 'cosine')
        else: # use label spreading
            self.vocabulary_embeddings = self.encoder.get_embeddings(text=self.vocabulary, save=False)
            X = np.concatenate([self.label_embeddings, self.vocabulary_embeddings], axis=0)
            y = np.concatenate([np.arange(len(self.label_embeddings)), 
                               np.zeros(len(self.vocabulary_embeddings), dtype=int)-1])
            
            lp = LabelPropagation(kernel='knn', gamma=20, n_neighbors=7, max_iter=10000, tol=0.001, n_jobs=4)
            lp.fit(X, y)
            distances = -lp.predict_proba(X)[len(self.label_embeddings):]

        closest_distance = np.min(distances, axis = 1)
        self.assigned_category = np.argmin(distances, axis = 1)

        if cutoff is not None: 
            mask = closest_distance <= cutoff
            # print(self.vocabulary.shape, self.assigned_category.shape,
            #       self.word_indicator_matrix.shape, mask.shape)
            self.keywords = self.vocabulary[mask.astype(bool)]
            self.assigned_category = self.assigned_category[mask.astype(bool)]
            self.word_indicator_matrix = self.word_indicator_matrix[:, np.where(mask)[0]]
        
        if topk is not None: 
            uniques = np.unique(self.assigned_category)
            mask = np.zeros(len(closest_distance), dtype=int)
            topk_assigned = np.copy(mask)

            _, counts = np.unique(self.assigned_category, return_counts=True)
            print('Found assigned category counts', counts)
            if min_topk==True:
                topk = np.min([topk, np.min(counts)])

            for u in uniques:
                u_inds = np.where(self.assigned_category==u)[0]
                u_dists = closest_distance[u_inds]
                sorted_inds = np.argsort(u_dists)[:topk]
                mask[u_inds[sorted_inds]] = 1
                topk_assigned[u_inds[sorted_inds]] = u

            self.keywords = self.vocabulary[mask.astype(bool)]
            self.assigned_category = self.assigned_category[mask.astype(bool)]
            self.word_indicator_matrix = self.word_indicator_matrix[:, np.where(mask)[0]]
        
        return self.keywords, self.assigned_category

    def create_label_matrix(self):
        self.word_indicator_matrix = np.where(self.word_indicator_matrix==0, -1, 0)
        for i, key in enumerate(self.keywords):
            self.word_indicator_matrix[:, i] = np.where(self.word_indicator_matrix[:, i]!= -1, self.assigned_category[i], -1)
        self.label_matrix = pd.DataFrame(self.word_indicator_matrix, columns = self.keywords)

    
