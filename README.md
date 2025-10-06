**Kannada Character Recognition using BeiT**

Deep learning implementation using BeiT (Bidirectional Encoder representation from Image Transformers) for recognizing 621 classes of handwritten Kannada characters (587 main aksharas + 34 ottaksharas).


**Overview**

This project implements a BeiT-based character recognition system for complete Kannada script coverage, achieving 99.88% validation accuracy. The model leverages self-supervised pre-training through masked image modeling and fine-tunes on 155,056 handwritten Kannada character images.
Results

Validation Accuracy: 99.88%
Training Accuracy: 99.99%
Precision/Recall/F1-Score: 0.9988
Generalization Gap: 0.11%
Main Aksharas Accuracy: 99.XX%
Ottaksharas Accuracy: 99.XX%
Training Time: ~5 hours (7 epochs on Tesla P100 GPU)
Model Size: 86M parameters

**Dataset**

Source: Handwritten Kannada Characters Dataset (Kaggle)
Total Images: 155,056
Classes: 621 total
Main Aksharas: 587 classes (146,556 images)
Ottaksharas: 34 classes (8,500 images)
Split: 80% training (124,044), 20% validation (31,012)
Split Strategy: Stratified sampling ensuring balanced class distribution
Image Size: 224×224 (resized and normalized)

**Model Architecture**

Base Model: BeiT Base (beit_base_patch16_224)
Pre-training Method: Self-supervised masked image modeling
Framework: PyTorch with timm library v0.9.x
Input Resolution: 224×224×3
Output: 621 classes (softmax)
Parameters: 86,239,533 (all trainable)


**Training Configuration**

pythonOptimizer: AdamW

Learning Rate: 5e-5 (initial, adjusted for BeiT)

Weight Decay: 0.05

Scheduler: CosineAnnealingLR
  - T_max: 7 epochs
  - Min LR: 1e-6
Batch Size: 32
Epochs: 7
Loss Function: CrossEntropyLoss
Device: CUDA (Tesla P100-PCIE-16GB)
