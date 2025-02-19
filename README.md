This repository contains three deep learning projects demonstrating **classification**, **regression**, and **image classification** tasks. Each project is implemented using TensorFlow/Keras and includes extensive logging with Weights & Biases (wandb) and TensorBoard integration where applicable.

## Overview

### 1. Classification Task (Breast Cancer Dataset)
- **Problem:** Binary classification using the Breast Cancer Wisconsin dataset.
- **Model:** A multi-layer perceptron (MLP).
- **Metrics:** Accuracy, precision, recall, F1 score (per class and overall), ROC and PR curves, per-class error analysis.
- **Artifacts:** Training history plots, model architecture visualization, and detailed classification reports.

Colab Code:

### 2. Regression Task (California Housing Dataset)
- **Problem:** Regression using the California Housing dataset.
- **Model:** A simple MLP for regression.
- **Metrics:** MSE, MAE, R² along with additional metrics (RMSE, Median Absolute Error, Explained Variance, MAPE). The continuous target is also binarized (using the median) to compute classification-like metrics (accuracy, precision, recall, F1) along with ROC and PR curves.
- **Artifacts:** Training history plots, model visualization, regression metrics, and error analysis.

Colab Code:

### 3. Image Classification Task (CIFAR-10)
- **Problem:** Multi-class image classification using the CIFAR-10 dataset.
- **Model:** A Convolutional Neural Network (CNN) with Batch Normalization and Dropout.
- **Metrics:** Overall and per-class accuracy, precision, recall, F1 score, ROC and PR curves (one-vs-rest), confusion matrix, and per-class error analysis.
- **Artifacts:** Training history plots, model architecture diagram, integration with TensorBoard, and detailed error analysis.
- **Additional Features:** Images are normalized, and TensorBoard is used for monitoring training logs.

Colab Code:

## Dependencies

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- Matplotlib
- Seaborn
- Weights & Biases (wandb)
- TensorBoard

Install the dependencies using:

```bash
pip install tensorflow scikit-learn matplotlib seaborn wandb tensorboard
```
YouTube link : 
