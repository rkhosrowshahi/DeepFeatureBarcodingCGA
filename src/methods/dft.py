import numpy as np

def dft_barcoding(features):
    return np.fft.fft(features).real > 0


class DFT:
    def __init__(self, num_features, *args, **kwargs):
        self.num_features = num_features

    def fit(self, train_features, train_labels):
        pass

    def transform(self, features):
        return dft_barcoding(features)