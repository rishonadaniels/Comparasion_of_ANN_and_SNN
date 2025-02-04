# 🔬 ANN vs. SNN: A Comparative Analysis  

This repository contains the implementation, results, and documentation for the project **"Comparison of Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs)."**  
The goal of this research is to analyze the **differences in architecture, training methods, efficiency, and real-world applications** between ANNs, SNNs, and **Vision Transformers (ViT)**.

---

## 📌 Project Overview  

### Why Compare ANN and SNN?
Artificial Neural Networks (ANNs) have been the foundation of deep learning for years, excelling in tasks such as **image classification, natural language processing, and reinforcement learning**. However, they come with high **computational costs and energy consumption**.  
Spiking Neural Networks (SNNs) attempt to **mimic biological neurons**, offering event-driven computation and energy efficiency, but they are harder to train and optimize.  
We also include **Vision Transformers (ViT)** to examine their effectiveness in modern image recognition.

This project aims to:
- Compare the performance of **ANNs, SNNs, CNNs, and ViTs** across different datasets.
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

## 📊 Key Features

1. **ANN, CNN, and SNN Implementations**:
- Includes training and evaluation scripts for feedforward ANNs, CNNs, and SNNs.
- SNN implementation supports surrogate gradient learning.

2. **Vision Transformer (ViT) Fine-Tuning**:
- Fine-tuned a pre-trained ViT model on CIFAR-10 for state-of-the-art accuracy.

3. **Comprehensive Performance Metrics**:
- Accuracy, training time, model size, and energy efficiency comparison across architectures.

4. **Detailed Results**:
- Includes visualizations, tables, and logs of experiments conducted on MNIST, Fashion-MNIST, and CIFAR-10.

---

## 🧪 How to Run

1. Train models:
- Example (training an ANN on MNIST):
  ```
  python scripts/train_ann.py --dataset mnist --epochs 20
  ```

2. Fine-tune ViT:




3. Visualize results:

---
## ⚡ Experiments & Results
We evaluated these models on the following datasets:
✅ MNIST (Handwritten digit classification)
✅ Fashion-MNIST (Clothing classification)
✅ CIFAR-10 (Object classification)

Comparison Table:

| Model  | Dataset         | Accuracy | Training Time | Model Size | Energy Efficiency |
|--------|---------------|----------|--------------|------------|------------------|
| ANN    | MNIST         | 97.5%    | 137.82 sec  | 0.39 MB    | High Power Usage |
| CNN    | MNIST         | 99.1%    | 62.81 sec   | 1.61 MB    | Moderate Power Usage |
| SNN    | MNIST         | 94.22%   | 638 sec     | 3.04 MB    | Low Power Usage |
| ANN    | Fashion-MNIST | 89.25%   | 294.02 sec  | 0.39 MB    | High Power Usage |
| CNN    | Fashion-MNIST | 93.08%   | 285.56 sec  | 3.69 MB    | Moderate Power Usage |
| SNN    | Fashion-MNIST | 86.2%    | 835.68 sec  | 3.04 MB    | Low Power Usage |
| ANN    | CIFAR-10      | 57.68%   | 286.66 sec  | 90.83 MB   | High Power Usage |
| ViT    | CIFAR-10      | 93.45%   | 8424.16 sec | 327.38 MB  | High Power Usage |


---

## 📜 Project Documentation
The full project report, including the methodology, results, and conclusion, is available in the `report/` directory.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the repository, feel free to fork it, make changes, and submit a pull request.

---



