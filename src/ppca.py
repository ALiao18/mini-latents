from base import LinearLatentModels
from noise_model import IsotropicNoise, AnisotropicNoise
from em_core import fit_em
import numpy as np

class ProbabilisticLinearLatentModels(LinearLatentModels):
    def __init__(self, n_components: None):
        super().__init__(n_components)
        self.Xc = None

    def transform(self, X):
        # get centered X
        self.mean_ = X.mean(axis=0)
        self.Xc = X - self.mean
        

    def fit(self):
        pass

class pPCA(ProbabilisticLinearLatentModels):
    def __init__(self):
        super().__init__()

class FA(ProbabilisticLinearLatentModels):
    def __init__(self):
        super().init__()

        
