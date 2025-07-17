"""
Example usage of Deep Feature Barcoding with Combinatorial Genetic Algorithm

This script demonstrates how to use the various barcoding methods
with synthetic data for testing and development purposes.
"""

import numpy as np
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import evaluate_retrieval, setup_seed
from src.methods import CGA, AHash, DHash, MinMax, DFT, ITQ, LSH


def generate_synthetic_data(n_samples=1000, n_features=100, n_classes=5, seed=42):
    """
    Generate synthetic dataset for testing barcoding methods.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features per sample
        n_classes: Number of classes
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(seed)
    
    # Generate random features with some class structure
    features = []
    labels = []
    
    for class_id in range(n_classes):
        # Generate class-specific mean
        class_mean = np.random.randn(n_features) * 2
        
        # Generate samples for this class
        n_class_samples = n_samples // n_classes
        if class_id < n_samples % n_classes:
            n_class_samples += 1
            
        class_features = np.random.randn(n_class_samples, n_features) + class_mean
        class_labels = np.full(n_class_samples, class_id)
        
        features.append(class_features)
        labels.append(class_labels)
    
    features = np.vstack(features)
    labels = np.hstack(labels)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(features))
    features = features[shuffle_idx]
    labels = labels[shuffle_idx]
    
    return features, labels


def split_data(features, labels, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Split data into train, validation, and test sets."""
    n_samples = len(features)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    train_features = features[:n_train]
    train_labels = labels[:n_train]
    
    val_features = features[n_train:n_train + n_val]
    val_labels = labels[n_train:n_train + n_val]
    
    test_features = features[n_train + n_val:]
    test_labels = labels[n_train + n_val:]
    
    return (train_features, train_labels, 
            val_features, val_labels, 
            test_features, test_labels)


def run_barcoding_example(method_name, barcoder, train_features, train_labels, 
                         test_features, test_labels, k=5):
    """
    Run a single barcoding method and evaluate performance.
    
    Args:
        method_name: Name of the method for display
        barcoder: Initialized barcoding method object
        train_features, train_labels: Training data
        test_features, test_labels: Test data
        k: Number of neighbors for evaluation
    
    Returns:
        Tuple of (f1_score, precision_k, mean_ap)
    """
    print(f"\n--- Running {method_name} ---")
    
    # Fit the method on training data
    print("Fitting barcoder on training data...")
    barcoder.fit(train_features, train_labels)
    
    # Transform data to binary codes
    print("Transforming data to binary codes...")
    train_binary = barcoder.transform(train_features)
    test_binary = barcoder.transform(test_features)
    
    print(f"Binary code shape: {test_binary.shape}")
    print(f"Code range: [{test_binary.min()}, {test_binary.max()}]")
    
    # Evaluate retrieval performance
    print("Evaluating retrieval performance...")
    f1, precision_k, mean_ap = evaluate_retrieval(
        query_codes=test_binary, 
        database_codes=train_binary,
        query_labels=test_labels, 
        database_labels=train_labels, 
        k=k
    )
    
    print(f"Results - F1: {f1:.4f}, Precision@{k}: {precision_k:.4f}, mAP: {mean_ap:.4f}")
    
    return f1, precision_k, mean_ap


def main():
    """Main example function."""
    print("Enhancing Image Retrieval Through Optimal Barcode Representation - Example Usage")
    print("=" * 50)
    
    # Set random seed for reproducibility
    setup_seed(42)
    
    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    features, labels = generate_synthetic_data(
        n_samples=1000, 
        n_features=64, 
        n_classes=5, 
        seed=42
    )
    
    print(f"Generated dataset: {features.shape[0]} samples, {features.shape[1]} features, {len(np.unique(labels))} classes")
    
    # Split data
    print("Splitting data into train/val/test sets...")
    train_features, train_labels, val_features, val_labels, test_features, test_labels = split_data(
        features, labels
    )
    
    print(f"Train: {len(train_features)} samples")
    print(f"Validation: {len(val_features)} samples") 
    print(f"Test: {len(test_features)} samples")
    
    # Initialize barcoding methods
    num_features = features.shape[1]
    num_bits = 32  # Using smaller number of bits for faster computation in example
    k = 5  # Number of neighbors for evaluation
    
    methods = []
    
    # Traditional methods (fast)
    methods.append(("aHash", AHash(num_features=num_features)))
    methods.append(("dHash", DHash(num_features=num_features)))
    methods.append(("MinMax", MinMax(num_features=num_features)))
    methods.append(("DFT", DFT(num_features=num_features)))
    
    # Learning-based methods
    methods.append(("ITQ", ITQ(num_features=num_features, num_bits=num_bits)))
    methods.append(("LSH", LSH(num_features=num_features, num_bits=num_bits)))
    
    # CGA methods (slower, but potentially better performance)
    # Note: Using small population and generations for faster demo
    methods.append(("CGA-dHash", CGA(
        num_features=num_features,
        crossover_prob=0.9,
        mutation_prob=0.1,
        n_gen=10,  # Reduced for demo
        pop_size=20,  # Reduced for demo
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        test_features=test_features,
        test_labels=test_labels,
        k=k,
        binarizer="diff"
    )))
    
    # Run all methods and collect results
    results = {}
    
    for method_name, barcoder in methods:
        try:
            f1, precision_k, mean_ap = run_barcoding_example(
                method_name, barcoder, 
                train_features, train_labels,
                test_features, test_labels,
                k=k
            )
            results[method_name] = {
                'f1': f1,
                'precision_k': precision_k,
                'mean_ap': mean_ap
            }
        except Exception as e:
            print(f"Error running {method_name}: {str(e)}")
            results[method_name] = None
    
    # Print summary results
    print("\n" + "=" * 50)
    print("SUMMARY RESULTS")
    print("=" * 50)
    print(f"{'Method':<15} {'F1':<8} {'Prec@{k}':<8} {'mAP':<8}".format(k=k))
    print("-" * 50)
    
    for method_name, result in results.items():
        if result is not None:
            print(f"{method_name:<15} {result['f1']:<8.4f} {result['precision_k']:<8.4f} {result['mean_ap']:<8.4f}")
        else:
            print(f"{method_name:<15} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}")
    
    print("\nExample completed successfully!")
    print("\nNote: This example uses synthetic data and reduced parameters for speed.")
    print("For real experiments, use larger populations, more generations, and real datasets.")


if __name__ == "__main__":
    main() 