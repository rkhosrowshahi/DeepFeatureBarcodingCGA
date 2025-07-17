import numpy as np
from sklearn.decomposition import PCA

# ITQ Implementation (Optimized LSH)
class ITQ:
    def __init__(self, num_features, num_bits):
        self.num_features = num_features
        self.num_bits = num_bits
        self.pca = PCA(n_components=self.num_bits)
        self.R = None  # Rotation matrix
 
    def fit(self, train_features, train_labels):
        # Step 1: Apply PCA
        pca_features = self.pca.fit_transform(train_features)
        # Step 2: Initialize random rotation matrix
        self.R = np.random.randn(self.num_bits, self.num_bits)
        self.R, _ = np.linalg.qr(self.R)  # Orthogonalize
        # Step 3: Iterative optimization
        for _ in range(20):  # 20 iterations as per ITQ paper
            Z = pca_features @ self.R
            B = np.sign(Z)  # Binary codes
            U, _, Vt = np.linalg.svd(B.T @ pca_features)
            self.R = Vt.T @ U.T
        return self
    
    def transform(self, features):
        pca_features = self.pca.transform(features)
        return np.sign(pca_features @ self.R).astype(np.uint8)