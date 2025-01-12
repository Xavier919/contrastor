# CONTRASTOR: Supervised Contrastive Learning of Document Embeddings with Siamese Transformers

## Overview
CONTRASTOR is a deep learning framework that applies supervised contrastive learning principles to derive document embeddings from high-dimensional textual data. The framework transforms text data into dense, 32-dimensional embeddings using a siamese transformer architecture. These embeddings can serve as input features for training standard machine learning classifiers like logistic regression, SVM, or neural networks for downstream tasks such as document classification.

## Architecture

### Siamese Transformer Design
The architecture comprises twin transformer encoders that share weights to process document pairs:
- Four attention heads per encoder
- Single transformer layer
- Feed-forward module with hidden dimension of 150
- Input processing capacity of 5000 tokens
- Output dimension of 32

### Distance and Loss Functions
The framework employs Euclidean distance to measure similarity between encoded document pairs. The contrastive loss function incorporates:
- Binary targets indicating document pair similarity (1) or dissimilarity (0)
- Predicted distances between sample pairs
- Margin parameter defining minimum separation between different classes

## Data Processing Pipeline

### Text Preprocessing
The pipeline implements standard text preprocessing steps:
1. Tokenization into word lists
2. Lowercase conversion
3. Removal of punctuation and newlines
4. Stopword filtering using SpaCy's French language model
5. Frequency-based token filtering (removal of tokens appearing in >99% or <1% of texts)
6. Length normalization to 5000 tokens

### Text Representation
Documents are encoded using:
- Pre-trained French Wiki Word Vectors
- Document matrix dimensions: 300x5000
  - 300: word vector dimensionality
  - 5000: normalized document length

## Training Implementation

### Configuration
- RMSprop optimizer with 0.001 learning rate
- Batch size of 8
- 5-fold cross-validation strategy
- Early stopping based on test accuracy
- Training performed on four Tesla V100-SXM2-16GB GPUs

### Classification Results
Using the learned embeddings as input features, an SVM classifier achieved:
- Precision: 0.941
- Recall: 0.942
- F1 score: 0.941
- Accuracy: 0.942