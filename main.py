import argparse
from src.utils import setup_seed, load_dataset, evaluate_retrieval
from src.methods import *

def main(args):
    # Set seed
    setup_seed(args.seed)
    # Load dataset
    train_features, train_labels, val_features, val_labels, test_features, test_labels = load_dataset(args.dataset, args.feature_selection, args.download)
    # Check number of features
    num_features = train_features.shape[1]
    if not (num_features == val_features.shape[1] and num_features == test_features.shape[1]):
        raise ValueError("Number of features in train, val, and test sets must be the same")
    # Check number of bits
    num_bits = args.n_bits

    # Select method
    barcoder = None
    if args.method == f"CGA-dHash":
        barcoder = CGA(num_features=num_features, crossover_prob=args.cga_crossover_rate, mutation_prob=args.cga_mutation_rate, 
                        n_gen=args.cga_n_gen, pop_size=args.cga_pop_size,
                        train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels,
                        test_features=test_features, test_labels=test_labels, k=args.k, binarizer="diff")
    elif args.method == f"CGA-DFT":
        barcoder = CGA(num_features=num_features, crossover_prob=args.cga_crossover_rate, mutation_prob=args.cga_mutation_rate, 
                        n_gen=args.cga_n_gen, pop_size=args.cga_pop_size,
                        train_features=train_features, train_labels=train_labels, val_features=val_features, val_labels=val_labels,
                        test_features=test_features, test_labels=test_labels, k=args.k, binarizer="dft")
    elif args.method == f"aHash":
        barcoder = AHash(num_features=num_features)
    elif args.method == f"dHash":
        barcoder = DHash(num_features=num_features)
    elif args.method == f"MinMax":
        barcoder = MinMax(num_features=num_features)
    elif args.method == f"DFT":
        barcoder = DFT(num_features=num_features)
    elif args.method == f"ITQ":
        barcoder = ITQ(num_features=num_features, num_bits=num_bits)
    elif args.method == f"LBP":
        barcoder = LBP(num_features=num_features, n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.method == f"LSH":
        barcoder = LSH(num_features=num_features, num_bits=num_bits)
    elif args.method == f"DHN":
        barcoder = HashingNN(num_features=num_features, num_bits=num_bits, loss_fn="dhn", n_epochs=args.n_epochs, device=args.device)
    elif args.method == f"DSH":
        barcoder = HashingNN(num_features=num_features, num_bits=num_bits, loss_fn="dsh", n_epochs=args.n_epochs, device=args.device)
    elif args.method == f"DTSH":
        barcoder = HashingNN(num_features=num_features, num_bits=num_bits, loss_fn="triplet", n_epochs=args.n_epochs, device=args.device)
    elif args.method == f"CSQ":
        barcoder = HashingNN(num_features=num_features, num_bits=num_bits, loss_fn="contrastive", n_epochs=args.n_epochs, device=args.device)
    elif args.method == f"DPSH":
        barcoder = HashingNN(num_features=num_features, num_bits=num_bits, loss_fn="pairwise", n_epochs=args.n_epochs, device=args.device)
    elif args.method == f"Quantization":
        barcoder = HashingNN(num_features=num_features, num_bits=num_bits, loss_fn="quantization", n_epochs=args.n_epochs, device=args.device)
    else:
        raise ValueError(f"Method {args.method} not found")

    # Fit train data with barcoder
    barcoder.fit(train_features, train_labels)
    # Transform test data into binary codes with barcoder
    train_binary = barcoder.transform(train_features)
    test_binary = barcoder.transform(test_features)

    # Evaluate retrieval performance
    f1, precision_k, mean_ap = evaluate_retrieval(query_codes=test_binary, database_codes=train_binary, 
                                                  query_labels=test_labels, database_labels=train_labels, k=args.k)
    
    print(f"{args.method} - F1 score: {f1:.4f}, Precision@k: {precision_k:.4f}, mAP: {mean_ap:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset selection
    parser.add_argument("--dataset", type=str, default="tcga_brain_kimianet", 
                        choices=["cifar10", "cifar100", "covid19", "fashion",
                                "tcga_brain_kimianet", "tcga_endocrine_kimianet", 
                                "tcga_gastrointestinal_kimianet", "tcga_gynecologic_kimianet", 
                                "tcga_liver_kimianet", "tcga_mesenchymal_kimianet", 
                                "tcga_pulmonary_kimianet", "tcga_urinary_tract_kimianet",
                                "tcga_brain_densenet121", "tcga_endocrine_densenet121", 
                                "tcga_gastrointestinal_densenet121", "tcga_gynecologic_densenet121", 
                                "tcga_liver_densenet121", "tcga_mesenchymal_densenet121", 
                                "tcga_pulmonary_densenet121", "tcga_urinary_tract_densenet121"],
                        help="Dataset to use")
    parser.add_argument("--feature_selection", default=False, action="store_true", help="Whether to perform feature selection")
    parser.add_argument("--download", default=False, action="store_true", help="Download datasets if they don't exist")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors to retrieve")
    # Method selection
    parser.add_argument("--method", type=str, default="CGA-dHash", 
                        choices=["CGA-dHash", "CGA-DFT", "LBP", "ITQ", "DHN", "DSH", "DTSH", 
                                 "CSQ", "DPSH", "Quantization", "LSH", "aHash", "dHash", "MinMax", "DFT"],
                        help="Method to use")
    # HashingNN parameters
    parser.add_argument("--n_bits", type=int, default=128, help="Number of bits to hash")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    # CGA parameters
    parser.add_argument("--cga_crossover_rate", type=float, default=0.9, help="Crossover rate")
    parser.add_argument("--cga_mutation_rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--cga_n_gen", type=int, default=100, help="Number of generations")
    parser.add_argument("--cga_pop_size", type=int, default=100, help="Population size")
    # Other configuration parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()
    main(args)