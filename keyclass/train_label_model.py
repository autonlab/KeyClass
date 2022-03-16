import logging
import numpy as np
import pandas as pd
import snorkel.labeling
from snorkel.labeling.model.label_model import LabelModel
from snorkel.labeling import LFAnalysis

logging.basicConfig(level=logging.INFO)


class LabelModelWrapper:
    """Class to train any weak supervision label model. 

        This class is an abstraction to the label models. We can
        ideally use any label model, but currently we only support
        data programing. Future plan is to include Dawid-Skeene.

        Parameters
        ---------- 
        y_train: np.array
            Gold training/development set labels

        n_classes: int
            Number of classes/categories. Default 2. 

        label_matrix: pd.DataFrame or np.array
            Label matrix of votes of each LF on all data points
    """

    def __init__(self, label_matrix, y_train=None, n_classes=2, 
        device='cuda', model_name='data_programming'):
        if not isinstance(label_matrix, pd.DataFrame):
            raise ValueError(f'label_matrix must be a DataFrame.')

        _VALID_LABEL_MODELS = ['data_programming']
        if model_name not in _VALID_LABEL_MODELS:
            raise ValueError(f'model_name must be one of {_VALID_LABEL_MODELS} but passed {model_name}.')
        
        self.label_matrix = label_matrix.to_numpy()
        self.y_train = y_train
        self.n_classes = n_classes 
        self.LF_names = list(label_matrix.columns)
        self.learned_weights = None # learned weights of the labeling functions 
        self.trained = False # The label model is not trained yet
        self.device=device
        self.model_name=model_name

    def display_LF_summary_stats(self):
        """Displays summary statistics for LFs
        """
        df_LFAnalysis = LFAnalysis(L = self.label_matrix).lf_summary(Y = self.y_train, 
            est_weights = self.learned_weights)
        df_LFAnalysis.index = self.LF_names
        
        return df_LFAnalysis

    def train_label_model(self, n_epochs=500, class_balance=None, 
        log_freq=100, lr=0.01, seed=13, cuda=False):
        """Train the label model

            Parameters
            ---------- 
            n_epochs: int
                The number of epochs to train (where each epoch is a single 
                optimization step), default is 100
            
            class_balance: list
                Each class’s percentage of the population, by default None

            log_freq: int
                Report loss every this many epochs (steps), default is 10

            lr: float
                Base learning rate (will also be affected by lr_scheduler choice 
                and settings), default is 0.01

            seed: int
                A random seed to initialize the random number generator with
        """
        print('Training the {}')
        self.label_model = LabelModel(cardinality=self.n_classes, device=self.device)
        if cuda==True:
            self.label_model = self.label_model.cuda()
        self.label_model.fit(self.label_matrix, n_epochs=n_epochs, 
                             class_balance=class_balance, log_freq=log_freq, 
                             lr=lr, seed=seed, optimizer='sgd')
        self.trained = True
        self.learned_weights = self.label_model.get_weights()


    def predict_proba(self):
        """Predict probabilistic labels P(Y | lambda)
        """
        if not self.trained: 
            print("Model must be trained before predicting probabilistic labels")
            return

        y_proba = pd.DataFrame(self.label_model.predict_proba(L=self.label_matrix), 
            columns=[f'Class {i}' for i in range(self.n_classes)])
        return y_proba

    def predict(self, tie_break_policy = 'random'):
        """Predict labels using the trained label model with ties broken according to policy.
        
            Parameters
            ---------- 
            tie_break_policy: str
                Policy to break ties when converting probabilistic labels to predictions. 
                Refer snorkel package for more details. 

        """
        if not self.trained: 
            print("Model must be trained before predicting labels")
            return 0

        y_pred = self.label_model.predict(L=self.label_matrix, 
            tie_break_policy=tie_break_policy)
        return y_pred
