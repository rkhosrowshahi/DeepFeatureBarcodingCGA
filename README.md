# Deep Feature Barcoding with Combinatorial Genetic Algorithm (CGA)

This repository contains the implementation for deep feature barcoding using Combinatorial Genetic Algorithm for medical image retrieval, particularly focused on TCGA (The Cancer Genome Atlas) datasets.

## Overview

This project implements various hashing and barcoding methods for efficient medical image retrieval, with a focus on histopathological images from cancer datasets. The main contribution is the use of Combinatorial Genetic Algorithm (CGA) to optimize feature ordering for improved barcoding performance.

### Key Features

- **Multiple Barcoding Methods**: CGA-dHash, CGA-DFT, LSH, ITQ, LBP, and various neural network-based approaches (DHN, DSH, DTSH, CSQ, DPSH)
- **Comprehensive Dataset Support**: TCGA cancer datasets across multiple organ systems (Brain, Endocrine, Gastrointestinal, Gynecologic, Liver, Mesenchymal, Pulmonary, Urinary Tract)
- **Multiple Feature Extractors**: Support for both KimiaNet and DenseNet121 features
- **Robust Evaluation**: F1-score, Precision@k, and mean Average Precision (mAP) metrics
- **Efficient Implementation**: Optimized Hamming distance computation and vectorized operations

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, for neural network methods)

### Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Dataset Setup

The code expects datasets to be organized in the `data/` directory. You have two options for setting up datasets:

#### Option 1: Automatic Download (Recommended)

Use the `--download` flag to automatically download datasets when needed:

```bash
python main.py --dataset cifar10 --download
```

#### Option 2: Manual Download

Download the datasets from the following links and extract them to the `./data` directory:

1. **CIFAR-10**: https://drive.google.com/uc?export=download&id=1nJHt0v1g8bxlXKcQUFzR3HL28rqeyW4m
2. **CIFAR-100**: https://drive.google.com/uc?export=download&id=1V6vBIX-MW8LGAl7W8cGZHbI4D8FhcoeY
3. **COVID-19**: https://drive.google.com/uc?export=download&id=1yUhCG__EJUKei4s5dtagldO0Blh6EFzt
4. **Fashion-MNIST**: https://drive.google.com/uc?export=download&id=1zAgS9-fGtnRNKY8tVOqMkS8yZHnkDDGB
5. **TCGA**: https://drive.google.com/uc?export=download&id=1jZCslLLz_jJ6lctlfutvMN4ljfNn2PQU

After downloading (either automatically or manually), your data directory structure should look like:

```
data/
├── tcga/
│   ├── kimiaNet_train_data.xlsx
│   ├── kimiaNet_validation_data.xlsx
│   ├── kimiaNet_test_data.xlsx
│   ├── AllKimiaPatches/
│   └── AllDensePatches/
├── cifar10/
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── test_features.npy
│   └── test_labels.npy
├── cifar100/
├── covid19/
└── fashion/
```

## Usage

### Basic Usage

Run the main script with default parameters:

```bash
python main.py
```

### Advanced Usage

#### Using CGA-dHash on TCGA Brain dataset:

```bash
python main.py --dataset tcga_brain_kimianet --method CGA-dHash --k 10 --cga_n_gen 100
```

#### Using neural network methods:

```bash
python main.py --dataset tcga_brain_kimianet --method DHN --n_bits 64 --n_epochs 100 --device cuda
```

#### With feature selection:

```bash
python main.py --dataset tcga_brain_kimianet --method CGA-dHash --feature_selection
```

#### With automatic dataset download:

```bash
python main.py --dataset cifar10 --method CGA-dHash --download
```

### Available Parameters

- `--dataset`: Dataset to use (default: `tcga_brain_kimianet`)
- `--method`: Barcoding method (default: `CGA-dHash`)
- `--k`: Number of nearest neighbors for evaluation (default: 10)
- `--n_bits`: Number of bits for hash codes (default: 128)
- `--feature_selection`: Enable feature selection (only works for TCGA)
- `--download`: Automatically download datasets if they don't exist
- `--cga_n_gen`: Number of generations for CGA (default: 100)
- `--cga_pop_size`: Population size for CGA (default: 100)
- `--device`: Device to use (`cuda` or `cpu`)

### Supported Methods

| Method | Type | Description |
|--------|------|-------------|
| CGA-dHash | Genetic Algorithm | Optimized difference hashing |
| CGA-DFT | Genetic Algorithm | Optimized DFT-based barcoding |
| aHash | Traditional | Average hashing |
| dHash | Traditional | Difference hashing |
| MinMax | Traditional | Min-max normalization |
| DFT | Traditional | Discrete Fourier Transform |
| ITQ | Learning-based | Iterative Quantization |
| LBP | Learning-based | Linear Binary Patterns |
| LSH | Learning-based | Locality Sensitive Hashing |
| DHN | Neural Network | Deep Hashing Network |
| DSH | Neural Network | Deep Supervised Hashing |
| DTSH | Neural Network | Deep Triplet Supervised Hashing |
| CSQ | Neural Network | Contrastive Supervised Quantization |
| DPSH | Neural Network | Deep Pairwise Supervised Hashing |

### Supported Datasets

#### TCGA Datasets (by Organ System)

- **Brain**: `tcga_brain_kimianet`, `tcga_brain_densenet121`
- **Endocrine**: `tcga_endocrine_kimianet`, `tcga_endocrine_densenet121`
- **Gastrointestinal**: `tcga_gastrointestinal_kimianet`, `tcga_gastrointestinal_densenet121`
- **Gynecologic**: `tcga_gynecologic_kimianet`, `tcga_gynecologic_densenet121`
- **Liver**: `tcga_liver_kimianet`, `tcga_liver_densenet121`
- **Mesenchymal**: `tcga_mesenchymal_kimianet`, `tcga_mesenchymal_densenet121`
- **Pulmonary**: `tcga_pulmonary_kimianet`, `tcga_pulmonary_densenet121`
- **Urinary Tract**: `tcga_urinary_tract_kimianet`, `tcga_urinary_tract_densenet121`

#### Other Datasets

- **CIFAR-10**: `cifar10`
- **CIFAR-100**: `cifar100`
- **COVID-19**: `covid19`
- **Fashion-MNIST**: `fashion`

## Methodology

### Combinatorial Genetic Algorithm (CGA)

The main contribution of this work is the application of Combinatorial Genetic Algorithm to optimize feature ordering for barcoding methods. CGA optimizes the permutation of features to improve the effectiveness of various barcoding techniques for medical image retrieval.

### Evaluation Metrics

The code evaluates retrieval performance using three key metrics:

1. **F1-Score**: Weighted F1-score based on majority voting of retrieved labels
2. **Precision@k**: Precision of the top-k retrieved items
3. **Mean Average Precision (mAP)**: Average precision across all queries

## Code Structure

```
src/
├── methods/
│   ├── cga.py          # Compact Genetic Algorithm implementation
│   ├── hashingnn.py    # Neural network hashing methods
│   ├── dhash.py        # Difference hashing
│   ├── dft.py          # DFT-based barcoding
│   ├── itq.py          # Iterative Quantization
│   ├── lbp.py          # Linear Binary Patterns
│   ├── lsh.py          # Locality Sensitive Hashing
│   ├── ahash.py        # Average hashing
│   └── minmax.py       # Min-max normalization
└── utils.py            # Utility functions for data loading and evaluation
```

## Reproducibility

To ensure reproducibility of results:

1. Set the random seed using `--seed` parameter
2. Use the same dataset splits (train/validation/test)
3. Use identical hyperparameters as specified in the paper
4. Ensure consistent hardware setup for neural network methods



## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{khosrowshahli2025enhancing,
  title={Enhancing Image Retrieval Through Optimal Barcode Representation},
  author={Khosrowshahli, Rasa and Kheiri, Farnaz and Bidgoli, Azam Asilian and Tizhoosh, H.R. and Makrehchi, Masoud and Rahnamayan, Shahryar},
  journal={Scientific Report},
  year={2025}
}
```

## Code Availability

This code is made available for peer review and reproducibility purposes as required by publication guidelines. The code allows readers to repeat the published results and is deposited with a DOI for permanent access.

[![DOI](https://img.shields.io/badge/DOI-10.24433%2FCO.0386311.v1-blue)](https://doi.org/10.24433/CO.0386311.v1)

**Project Developer**: Rasa Khosrowshahli, Faculty of Mathematics and Science, Brock University

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Data Availability

The TCGA datasets used in this study are publicly available through the Cancer Genome Atlas Research Network. Processed features and dataset splits used in our experiments can be made available upon reasonable request, subject to data use agreements.

## Requirements

See `requirements.txt` for a complete list of dependencies.

## Contributing

This code is provided primarily for reproducibility of published results. For questions or issues, please open an issue on this repository.

## Acknowledgments

- The Cancer Genome Atlas (TCGA) Research Network for providing the medical image datasets
- KimiaNet and DenseNet121 for pre-trained feature extractors
- PyTorch and scikit-learn communities for excellent machine learning libraries

---

**Note**: This code is provided as-is for research reproducibility. Please ensure you have appropriate computational resources and data access permissions before running experiments.
