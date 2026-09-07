from .base import LinearLatentModels
from .noise_model import NoiseModel, IsotropicNoise, AnisotropicNoise
from abc import ABC, abstractmethod
from .em_core import EM
import numpy as np

class ProbabilisticLinearLatentModels(LinearLatentModels):
    def __init__(self, n_components: None):
        super().__init__(n_components)
        self.noise_model = NoiseModel()
        self.Xc = None

    def transform(self, X):
        # get centered X
        self.mean_ = X.mean(axis=0)
        self.Xc = X - self.mean

    @abstractmethod
    def fit(self):
        '''
        Fit the model using EM algorithm
        '''

class pPCA(ProbabilisticLinearLatentModels):
    def __init__(self): self.noise_model = IsotropicNoise()

    def fit(self, X):
        '''
        EM implementation
        '''

class FA(ProbabilisticLinearLatentModels):
    def __init__(self): self.noise_model = AnisotropicNoise()

    def fit(self, X):
        '''
        EM implementation
        '''

        
