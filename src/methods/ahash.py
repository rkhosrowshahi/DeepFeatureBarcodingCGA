import numpy as np

def ahash_barcoding(features):
    """Threshold-Based: Binarize using mean as threshold."""
    mean_vals = features.mean(axis=0)
    binary = (features > mean_vals).astype(np.uint8)
    return binary


class AHash:
    def __init__(self, num_features, *args, **kwargs):
        self.num_features = num_features

    def fit(self, train_features, train_labels):
        pass

    def transform(self, features):
        return ahash_barcoding(features)
