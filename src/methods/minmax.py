import numpy as np

def minmax_barcoding(features):
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    binary = (features > (min_vals + max_vals) / 2).astype(np.uint8)
    return binary


class MinMax:
    def __init__(self, num_features, *args, **kwargs):
        self.num_features = num_features

    def fit(self, train_features, train_labels):
        pass

    def transform(self, features):
        return minmax_barcoding(features)