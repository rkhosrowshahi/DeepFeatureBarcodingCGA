import torch
import os
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
import gdown
import zipfile
from pathlib import Path

# Hamming distance for binary codes
def hamming_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the Hamming distance between two sets of binary codes.
    
    Args:
        x (np.ndarray): First set of binary codes
        y (np.ndarray): Second set of binary codes
        
    Returns:
        np.ndarray: Hamming distances
    """
    return np.sum(x != y, axis=-1)

# Retrieval evaluation
def evaluate_retrieval(
    query_codes: np.ndarray,
    database_codes: np.ndarray,
    query_labels: np.ndarray,
    database_labels: np.ndarray,
    k: int = 5
) -> tuple[float, float, float]:
    """
    Evaluate retrieval performance using Hamming distance with F1, Precision@k, and mAP.
    Optimized with vectorized operations.
    """
    # Input validation
    if not all(isinstance(x, np.ndarray) for x in [query_codes, database_codes, query_labels, database_labels]):
        raise TypeError("All inputs must be numpy arrays")
    if len(query_codes) != len(query_labels):
        raise ValueError("Number of query codes and labels must match")
    if len(database_codes) != len(database_labels):
        raise ValueError("Number of database codes and labels must match")
    if query_codes.shape[1] != database_codes.shape[1]:
        raise ValueError("Code dimensions must match")
    if k < 1 or k > len(database_codes):
        raise ValueError(f"k must be between 1 and {len(database_codes)}")

    # Pre-convert to uint8
    query_codes = query_codes.astype(np.uint8)
    database_codes = database_codes.astype(np.uint8)

    # Compute Hamming distances efficiently using XOR
    distances = np.bitwise_xor(
        query_codes[:, np.newaxis, :], 
        database_codes[np.newaxis, :, :]
    ).sum(axis=2)

    # Get top-k indices
    top_k_indices = np.argpartition(distances, k, axis=1)[:, :k]
    retrieved_labels = database_labels[top_k_indices]

    # F1 Score (original majority voting)
    n_queries = len(query_labels)
    predicted_labels = np.zeros(n_queries, dtype=query_labels.dtype)
    for i in range(n_queries):
        # Ensure retrieved labels are 1D array of non-negative integers
        labels = retrieved_labels[i].flatten().astype(np.int64)
        counts = np.bincount(labels)
        predicted_labels[i] = counts.argmax()
    f1 = f1_score(query_labels, predicted_labels, average="weighted")

    # Precision@k
    # Calculate matches for each query
    matches = np.zeros((n_queries, k))
    for i in range(n_queries):
        # Compare each retrieved label with the query label
        matches[i] = (retrieved_labels[i] == query_labels[i]).astype(float)
    precisions_at_k = np.mean(matches, axis=1)
    precision_k = np.mean(precisions_at_k)

    # mAP
    sorted_indices = np.argsort(distances, axis=1)
    retrieved_labels_full = database_labels[sorted_indices]
    relevant = (retrieved_labels_full == query_labels[:, np.newaxis]).astype(int)
    precisions = np.cumsum(relevant, axis=1) / (np.arange(relevant.shape[1]) + 1)
    ap = np.sum(precisions * relevant, axis=1) / np.maximum(np.sum(relevant, axis=1), 1)
    mean_ap = np.mean(ap)

    return f1, precision_k, mean_ap

def load_data_from_excel(excel_link: str, dict_labels: dict, fl_p1: str, fl_p2: str):
    # fl_p1 = 'DenseNet121_features'
    # fl_p2 = '_DN121_features_dict'
    # excel_link = "kimiaNet_train_data.xlsx"

    df = pd.read_excel(f'./data/tcga/{excel_link}')
    missed_counter = 0
    total_counter = 0
    X_data, y_data = [], []
    for i in range(df.shape[0]):
        row = df.iloc[i]
        if not row['project_id'] in dict_labels.keys():
            
            continue
        try:
            file_link = f'./data/tcga/{fl_p1}/' + row.slide_name + f'_{fl_p2}.pickle'
            pickle_file = pickle.load(open(file_link, 'rb'))
            total_counter += 1
        except:
            missed_counter += 1
            continue

        fv = np.array(list(pickle_file.values()))
        if len(fv) == 0:
            continue
        mean_fv = np.mean(fv, axis=0)
        X_data.append(mean_fv)

        label = dict_labels[row.project_id]
        y_data.append(label)

    print(f'Missed {missed_counter} slides out of {total_counter + missed_counter}')
    X_data = np.array(X_data)
    y_data = np.array(y_data)

    return X_data, y_data



def load_data_from_numpy(dataset_name):

    train_features, train_labels = None, None
    test_features, test_labels = None, None
    val_features, val_labels = None, None

    # Load training data
    train_features = np.load(f"data/{dataset_name}/train_features.npy")
    train_labels = np.load(f"data/{dataset_name}/train_labels.npy")

    # Load test data
    test_features = np.load(f"data/{dataset_name}/test_features.npy")
    test_labels = np.load(f"data/{dataset_name}/test_labels.npy")

    val_features = train_features.copy()
    val_labels = train_labels.copy()

    return train_features, train_labels, val_features, val_labels, test_features, test_labels


def load_dataset(dataset_name: str, feature_selection: bool = False, download: bool = False, data_dir: str = './data'):
    # Check if dataset exists or attempt auto-download
    if not download_dataset(dataset_name, data_dir, download):
        raise FileNotFoundError(f"Dataset '{dataset_name}' not found and could not be downloaded. "
                               f"Please download manually or use --download flag.")
    
    X_train, y_train, X_val, y_val, X_test, y_test = None, None, None, None, None, None
    if dataset_name == "tcga_brain_kimianet": # Brain
        dict_labels = {'TCGA-GBM': 0, 'TCGA-LGG': 1}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")

        if feature_selection:
            selected_features = [887, 874, 900]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_endocrine_kimianet": # Endocrine
        dict_labels = {'TCGA-ACC': 0, 'TCGA-PCPG': 1, 'TCGA-THCA': 2}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")

        if feature_selection:
            selected_features = [662, 698, 695, 595, 736, 701, 999, 867, 566, 987, 609]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_gastrointestinal_kimianet": # Gastrointestinal
        dict_labels = {'TCGA-COAD': 0, 'TCGA-READ': 1, 'TCGA-ESCA': 2, 'TCGA-STAD': 3}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")

        if feature_selection:
            selected_features = [865, 872, 896, 934, 824, 828, 929, 850, 776, 867, 795]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_gynecologic_kimianet": # Gynecologic
        dict_labels = {'TCGA-CESC': 0, 'TCGA-OV': 1, 'TCGA-UCS': 2}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")

        if feature_selection:
            selected_features = [709, 942, 699, 732, 788, 908, 896, 863, 959, 781, 872, 727]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_liver_kimianet": # Liver
        dict_labels = {'TCGA-CHOL': 0, 'TCGA-LIHC': 1, 'TCGA-PAAD': 2}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")

        if feature_selection:
            selected_features = [791, 594, 830, 822, 936, 571, 1011, 562, 865, 807, 1019, 720]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_mesenchymal_kimianet": # Mesenchymal
        dict_labels = {'TCGA-UVM': 0, 'TCGA-SKCM': 1}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
    
        if feature_selection:
            selected_features = [938, 881]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_pulmonary_kimianet": # Pulmonary
        dict_labels = {'TCGA-LUAD': 0, 'TCGA-LUSC': 1, 'TCGA-MESO': 2}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")

        if feature_selection:
            selected_features = [1014, 1021, 986, 901, 997]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_urinary_tract_kimianet": # Urinary tract
        dict_labels = {"TCGA-BLCA": 0, "TCGA-KICH": 1, "TCGA-KIRC": 2, "TCGA-KIRP": 3}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllKimiaPatches", fl_p2="KimiaNet_features_dict")

        if feature_selection:
            selected_features = [1013, 895, 875, 1006, 754, 978, 1015, 827]
            selected_features = np.sort(np.array(selected_features)) - 1
            X_train = X_train[:, selected_features]
            X_val = X_val[:, selected_features]
            X_test = X_test[:, selected_features]
    elif dataset_name == "tcga_brain_densenet121": # Brain
        dict_labels = {'TCGA-GBM': 0, 'TCGA-LGG': 1}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    elif dataset_name == "tcga_endocrine_densenet121": # Endocrine
        dict_labels = {'TCGA-ACC': 0, 'TCGA-PCPG': 1, 'TCGA-THCA': 2}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    elif dataset_name == "tcga_gastrointestinal_densenet121": # Gastrointestinal
        dict_labels = {'TCGA-COAD': 0, 'TCGA-READ': 1, 'TCGA-ESCA': 2, 'TCGA-STAD': 3}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    elif dataset_name == "tcga_gynecologic_densenet121": # Gynecologic
        dict_labels = {'TCGA-CESC': 0, 'TCGA-OV': 1, 'TCGA-UCS': 2}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    elif dataset_name == "tcga_liver_densenet121": # Liver
        dict_labels = {'TCGA-CHOL': 0, 'TCGA-LIHC': 1, 'TCGA-PAAD': 2}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    elif dataset_name == "tcga_mesenchymal_densenet121": # Mesenchymal
        dict_labels = {'TCGA-UVM': 0, 'TCGA-SKCM': 1}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    elif dataset_name == "tcga_pulmonary_densenet121": # Pulmonary
        dict_labels = {'TCGA-LUAD': 0, 'TCGA-LUSC': 1}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DenseNet121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DenseNet121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DenseNet121_features_dict")

        dict_labels = {'TCGA-MESO': 2}
        X_train1, y_train1 = load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels,
                                    fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test1, y_test1 = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels,
                                fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val1, y_val1 = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels,
                                fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_train=np.concatenate([X_train, X_train1], axis=0)
        y_train=np.concatenate([y_train, y_train1], axis=0)
        X_test=np.concatenate([X_test, X_test1], axis=0)
        y_test=np.concatenate([y_test, y_test1], axis=0)
        X_val=np.concatenate([X_val, X_val1], axis=0)
        y_val=np.concatenate([y_val, y_val1], axis=0)
    elif dataset_name == "tcga_urinary_tract_densenet121": # Urinary tract
        dict_labels = {"TCGA-BLCA": 0, "TCGA-KICH": 1, "TCGA-KIRC": 2, "TCGA-KIRP": 3}
        X_train, y_train  =  load_data_from_excel(excel_link="kimiaNet_train_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_val, y_val = load_data_from_excel(excel_link="kimiaNet_validation_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
        X_test, y_test = load_data_from_excel(excel_link="kimiaNet_test_data.xlsx", dict_labels=dict_labels, fl_p1="AllDensePatches", fl_p2="DN121_features_dict")
    elif dataset_name == "covid19":
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_numpy(dataset_name)
    elif dataset_name == "cifar10":
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_numpy(dataset_name)
    elif dataset_name == "cifar100":
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_numpy(dataset_name)
    elif dataset_name == "fashion":
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_numpy(dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not found")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None
    

def load_checkpoint(device, save_path, pruning, filename="checkpoint.pth.tar"):
    filepath = os.path.join(save_path, str(pruning) + filename)
    if os.path.exists(filepath):
        print("Load checkpoint from:{}".format(filepath))
        return torch.load(filepath, device)
    print("Checkpoint not found! path:{}".format(filepath))
    return None

def get_balanced_subset(dataset, num_samples, num_classes):
    samples_per_class = num_samples // num_classes
    if num_samples % num_classes > 0:
        samples_per_class += 1
    # Create an empty list to store the balanced dataset
    balanced_indices = []
    # Randomly select samples from each class for the training dataset
    for i in range(num_classes):
        class_indices = np.where(np.array(dataset.targets) == i)[0]
        selected_indices = np.random.choice(
            class_indices, samples_per_class, replace=False
        )
        balanced_indices.append(selected_indices)
    return np.asarray(balanced_indices).astype(int)


def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Dataset download configuration
DATASET_URLS = {
    'cifar10': 'https://drive.google.com/uc?export=download&id=1nJHt0v1g8bxlXKcQUFzR3HL28rqeyW4m',
    'cifar100': 'https://drive.google.com/uc?export=download&id=1V6vBIX-MW8LGAl7W8cGZHbI4D8FhcoeY',
    'covid19': 'https://drive.google.com/uc?export=download&id=1yUhCG__EJUKei4s5dtagldO0Blh6EFzt',
    'fashion': 'https://drive.google.com/uc?export=download&id=1zAgS9-fGtnRNKY8tVOqMkS8yZHnkDDGB',
    'tcga': 'https://drive.google.com/file/d/1jZCslLLz_jJ6lctlfutvMN4ljfNn2PQU/view?usp=sharing'
}

def check_dataset_exists(dataset_name: str, data_dir: str = './data') -> bool:
    """
    Check if a dataset exists in the data directory.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory where datasets are stored
        
    Returns:
        bool: True if dataset exists, False otherwise
    """
    data_path = Path(data_dir)
    
    if dataset_name.startswith('tcga_'):
        # For TCGA datasets, check if the tcga directory and required files exist
        tcga_path = data_path / 'tcga'
        required_files = [
            'kimiaNet_train_data.xlsx',
            'kimiaNet_validation_data.xlsx', 
            'kimiaNet_test_data.xlsx'
        ]
        required_dirs = ['AllKimiaPatches', 'AllDensePatches']
        
        if not tcga_path.exists():
            return False
            
        for file in required_files:
            if not (tcga_path / file).exists():
                return False
                
        for dir_name in required_dirs:
            if not (tcga_path / dir_name).exists():
                return False
                
        return True
    else:
        # For other datasets, check if the dataset directory and .npy files exist
        dataset_path = data_path / dataset_name
        required_files = [
            'train_features.npy',
            'train_labels.npy',
            'test_features.npy',
            'test_labels.npy'
        ]
        
        if not dataset_path.exists():
            return False
            
        for file in required_files:
            if not (dataset_path / file).exists():
                return False
                
        return True


def download_and_extract_dataset(dataset_name: str, data_dir: str = './data', force_download: bool = False) -> bool:
    """
    Download and extract a dataset from Google Drive.
    
    Args:
        dataset_name: Name of the dataset to download
        data_dir: Directory to store datasets
        force_download: If True, download even if dataset exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Map dataset names to base names for URL lookup
    base_dataset_name = dataset_name
    if dataset_name.startswith('tcga_'):
        base_dataset_name = 'tcga'
    
    if base_dataset_name not in DATASET_URLS:
        print(f"Error: No download URL configured for dataset '{dataset_name}'")
        return False
    
    # Check if dataset already exists
    if not force_download and check_dataset_exists(dataset_name, data_dir):
        print(f"Dataset '{dataset_name}' already exists. Skipping download.")
        return True
    
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    url = DATASET_URLS[base_dataset_name]
    print(f"Downloading dataset '{dataset_name}' from Google Drive...")
    
    try:
        # Download the file
        if 'drive.google.com/file/d/' in url:
            # Extract file ID from sharing URL
            file_id = url.split('/file/d/')[1].split('/')[0]
            download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        else:
            download_url = url
            
        # Download to temporary file
        temp_file = data_path / f'{base_dataset_name}_temp.zip'
        gdown.download(download_url, str(temp_file), quiet=False)
        
        # Extract the zip file
        print(f"Extracting dataset '{dataset_name}'...")
        with zipfile.ZipFile(temp_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        
        # Remove temporary file
        temp_file.unlink()
        
        print(f"Successfully downloaded and extracted dataset '{dataset_name}'")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset '{dataset_name}': {str(e)}")
        return False


def download_dataset(dataset_name: str, data_dir: str = './data', 
                         download: bool = False, force_download: bool = False) -> bool:
    """
    Automatically download dataset if it doesn't exist and auto_download is enabled.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Directory where datasets are stored
        auto_download: If True, automatically download missing datasets
        force_download: If True, download even if dataset exists
        
    Returns:
        bool: True if dataset is available (exists or successfully downloaded)
    """
    if check_dataset_exists(dataset_name, data_dir) and not force_download:
        return True
    
    if not download:
        print(f"Dataset '{dataset_name}' not found in '{data_dir}'.")
        print("Please download it manually or use --download flag.")
        print(f"Download links are available in the README.md file.")
        return False
    
    print(f"Dataset '{dataset_name}' not found. Attempting automatic download...")
    return download_and_extract_dataset(dataset_name, data_dir, force_download)