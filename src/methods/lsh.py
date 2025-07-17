import numpy as np

class LSH:
    def __init__(self, num_features, num_bits):
        self.num_features = num_features
        self.num_bits = num_bits
        self.projections = None

    def fit(self, train_features, train_labels, *args, **kwargs):
        self.projections = np.random.randn(self.num_features, self.num_bits) / np.sqrt(self.num_features)

    def transform(self, features):
        proj = features @ self.projections
        thresholds = np.median(proj, axis=0)
        return (proj > thresholds).astype(np.uint8)
    