import numpy as np

def dhash_barcoding(features: np.ndarray) -> np.ndarray:
    """Generate binary barcodes by comparing adjacent feature values in a circular manner.

    Args:
        features (np.ndarray): Input feature matrix of shape (n_samples, n_features).

    Returns:
        np.ndarray: Binary codes of shape (n_samples, n_features).
    """
    # Ensure we have at least 2 features
    if features.shape[1] < 2:
        raise ValueError("Features must have at least 2 dimensions")
    
    # Compare adjacent features
    binary = (features[:, :-1] < features[:, 1:]).astype(np.uint8)
    binary = np.hstack([binary, (features[:, 0] < features[:, -1]).reshape(-1, 1).astype(np.uint8)]).astype(np.uint8)
    
    return binary

class DHash:
    def __init__(self, num_features, *args, **kwargs):
        self.num_features = num_features

    def fit(self, train_features, train_labels):
        pass

    def transform(self, features):
        return dhash_barcoding(features)