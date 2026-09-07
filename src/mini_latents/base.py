import numpy as np
from abc import ABC, abstractmethod

class LinearLatentModels(ABC):
    def __init__(self, n_components: int):
        self.n_components           = n_components
        self.components_            = None # 2D array: eigenvectors of the top n_components
        self.mean_                  = None 
        self.noise_cov_             = None
        self.explained_variance_    = None # 1D array: eigenvalues of the top n_components 
        self.noise_variance_        = None
        self.Z                      = None

    @abstractmethod
    def fit(self, X: np.ndarray):
        '''
        Fit the model to the data X
        '''

    @abstractmethod
    def transform(self, X: np.ndarray):
        '''
        Transform the data X to the latent space
        '''

    def inverse_transform(self):
        '''
        Transform the latent representation Z back to the original space
        '''
        return self.Z @ self.components_.T + self.mean_

    def sample(self, n_samples: int):
        '''
        Sample from the model
        '''