# 🔬 ANN vs. SNN: A Comparative Analysis  

This repository contains the implementation, results, and documentation for the project **"Comparison of Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs)."**\
The goal of this research is to analyze the **differences in architecture, training methods, efficiency, and real-world applications** between ANNs, SNNs, Vision Transformers (ViT), and a **Hybrid ViT-SNN model**.


---

## 📌 Project Overview  

### Why Compare ANN and SNN?
Artificial Neural Networks (ANNs) have been the foundation of deep learning for years, excelling in tasks such as **image classification, natural language processing, and reinforcement learning**. However, they come with high **computational costs and energy consumption**.\
Spiking Neural Networks (SNNs) attempt to **mimic biological neurons**, offering event-driven computation and energy efficiency, but they are harder to train and optimize.\
We also include **Vision Transformers (ViT)** and a **Hybrid ViT-SNN Model**, examining their effectiveness in modern image recognition and energy-efficient AI.

This project aims to:

- Compare the performance of **ANNs, SNNs, CNNs, ViTs, and Hybrid ViT-SNN** across different datasets.
- Evaluate their **accuracy, training time, model size, and energy efficiency**.
- Provide insights into the **suitability of each architecture** for different AI applications.

---

## 📁 Repository Structure  

```plaintext
├── models/                  # ANN, SNN, CNN, and ViT model implementations
└── README.md
```
---

## 🚀 Getting Started

### Prerequisites
- Python >= 3.8
- Required Python libraries:
```plaintext
pip install numpy pandas matplotlib torch torchvision time snntorch transformers
```



### Installation
1. Clone the repository:



2. Install dependencies:



---

## 🏗 Model Architectures Implemented

### 🔹 Artificial Neural Network (ANN)

- Fully connected layers trained using **backpropagation** and **gradient descent**.
- Suitable for classification tasks but computationally intensive.

### 🔹 Convolutional Neural Network (CNN)

- Uses convolutional layers for **feature extraction**.
- Outperforms ANNs on image-based tasks like **MNIST and Fashion-MNIST**.

### 🔹 Spiking Neural Network (SNN)

- Uses **spike-based computation** and **event-driven processing**.
- Trained using **Surrogate Gradient Descent**.
- **Energy-efficient** but requires specialized training techniques.

### 🔹 Vision Transformer (ViT)

- A **pre-trained transformer-based model** fine-tuned on CIFAR-10.
- Uses **self-attention** instead of convolutions, achieving **state-of-the-art accuracy**.

### 🔹 Hybrid ViT-SNN Model

- Combines **ViT with SNN** by replacing activation functions (ReLU/GELU) with **Leaky Integrate-and-Fire (LIF) spiking neurons**.
- Extracts features using ViT while leveraging the energy efficiency of SNNs.
- Trained using surrogate gradient descent and membrane potential reset techniques.

---

## ⚡ Experiments & Results
We evaluated these models on the following datasets:
✅ MNIST (Handwritten digit classification)
✅ Fashion-MNIST (Clothing classification)
✅ CIFAR-10 (Object classification)

Comparison Table:

| Model  | Dataset         | Accuracy | Training Time | Model Size |
|--------|---------------|----------|--------------|------------|
| ANN    | MNIST         | 97.5%    | 137.82 sec  | 0.39 MB    |
| CNN    | MNIST         | 99.1%    | 62.81 sec   | 1.61 MB    |
| SNN    | MNIST         | 94.22%   | 638 sec     | 3.04 MB    |
| Vit    | MNIST         | 99.28%   | 4237.52 sec | 327.38 MB  |
| ANN    | Fashion-MNIST | 89.25%   | 294.02 sec  | 0.39 MB    |
| CNN    | Fashion-MNIST | 93.08%   | 285.56 sec  | 3.69 MB    |
| SNN    | Fashion-MNIST | 86.2%    | 835.68 sec  | 3.04 MB    |
| Vit    | Fashion-MNIST | 90.01%   | 394.52  sec | 327.38 MB  |
| ANN    | CIFAR-10      | 57.68%   | 286.66 sec  | 90.83 MB   |
| ViT    | CIFAR-10      | 93.45%   | 8424.16 sec | 327.38 MB  |
| ViTSNN | CIFAR-10      | 73.38%   | 12267.26 sec| 21.15 MB   |


---

## 📜 Project Documentation
The full project report, including the methodology, results, and conclusion, is available in the `report/` directory.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the repository, feel free to fork it, make changes, and submit a pull request.

---



