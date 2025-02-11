# 🔬 ANN vs. SNN: A Comparative Analysis  

This repository contains the implementation, results, and documentation for the project **"Comparison of Artificial Neural Networks (ANNs) and Spiking Neural Networks (SNNs)."**\
The goal of this research is to analyze the **differences in architecture, training methods, efficiency, and real-world applications** between ANNs, SNNs, Vision Transformers (ViT), and a **Hybrid ViT-SNN model**.

![A detailed diagram illustrating the architectures of Artificial Neural Networks (ANN), Spiking Neural Networks (SNN), Vision Transformers (ViT), and a](https://github.com/user-attachments/assets/6c8441fc-ae7e-41a0-ae79-b72ff307d75e)

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
📂 ann-vs-snn-comparison
 ┣ 📂 models         # ANN, SNN, CNN, ViT, Hybrid ViT-SNN implementations
 ┃ ┣ 📂 ANN          # ANN models (CIFAR-10, Fashion-MNIST, MNIST)
 ┃ ┣ 📂 CNN          # CNN models (CIFAR-10, Fashion-MNIST, MNIST)
 ┃ ┣ 📂 SNN          # SNN models (CIFAR-10, Fashion-MNIST, MNIST)
 ┃ ┣ 📂 ViT          # Vision Transformer models (CIFAR-10, Fashion-MNIST, MNIST)
 ┃ ┣ 📂 ViTSNN       # Hybrid ViT-SNN model
 ┣ 📂 docs          # Documentation and reports
 ┃ ┣ 📜 Project Report.pdf   # Main project report
 ┃ ┣ 📜 Presentation.pdf  # A presentation of the project
 ┣ 📜 README.md      # This document
```

---

## 🚀 Getting Started

### Prerequisites
- Python >= 3.8
- Required Python libraries:
```plaintext
pip install numpy torch torchvision time snntorch transformers os
```



## 🔧 Installation & Setup


### Clone the repository

```bash
git clone https://github.com/rishonadaniels/Comparasion_of_ANN_and_SNN.git
cd ann-vs-snn-comparison
```

---

## 🏗 Model Architectures Implemented

### 🔹 Artificial Neural Network (ANN)

- Fully connected layers trained using **backpropagation** and **gradient descent**.
- Suitable for classification tasks but computationally intensive.

  ![The-Architecture-of-a-Neural-Network](https://github.com/user-attachments/assets/5fe0ac4c-299f-4619-91c8-067167c49c61)


### 🔹 Convolutional Neural Network (CNN)

- Uses convolutional layers for **feature extraction**.
- Outperforms ANNs on image-based tasks like **MNIST and Fashion-MNIST**.

  ![1680532048475](https://github.com/user-attachments/assets/990efd62-ec5b-477c-9b9b-fe4b31b864cd)


### 🔹 Spiking Neural Network (SNN)

- Uses **spike-based computation** and **event-driven processing**.
- Trained using **Surrogate Gradient Descent**.
- **Energy-efficient** but requires specialized training techniques.

  ![SNN-architectures](https://github.com/user-attachments/assets/53df3b57-b481-4464-9f99-466496824736)


### 🔹 Vision Transformer (ViT)

- A **pre-trained transformer-based model** fine-tuned on CIFAR-10.
- Uses **self-attention** instead of convolutions, achieving **state-of-the-art accuracy**.

  ![ 21 52 47](https://github.com/user-attachments/assets/a5c943b3-7bb7-4093-947a-a7c4fb10ce1b)


### 🔹 Hybrid ViT-SNN Model

- Combines **ViT with SNN** by replacing activation functions (ReLU/GELU) with **Leaky Integrate-and-Fire (LIF) spiking neurons**.
- Extracts features using ViT while leveraging the energy efficiency of SNNs.
- Trained using surrogate gradient descent and membrane potential reset techniques.

---

## 🔬 How to Run the Experiments

### **Run Jupyter Notebooks**
To train and evaluate models, open the appropriate notebook in the `models/` directory:

```bash
jupyter notebook
```
Then, open one of the following notebooks:

#### **ANN Models:**
- `models/ANN/Feedforward_ANN_perceptron_CIFAR.ipynb`
- `models/ANN/Feedforward_ANN_perceptron_Fashion_MNIST.ipynb`
- `models/ANN/Feedforward_ANN_perceptron_MNIST.ipynb`

#### **CNN Models:**
- `models/CNN/CNN_CIFAR10.ipynb`
- `models/CNN/CNN_perceptron_Fashion_MNIST.ipynb`
- `models/CNN/CNN_perceptron_MNIST.ipynb`

#### **SNN Models:**
- `models/SNN/Feedforward_SNN_MNIST.ipynb`
- `models/SNN/Feedforward_SNN_Fashion_MNIST.ipynb`

#### **ViT Models:**
- `models/ViT/ViT_CIFAR10.ipynb`
- `models/ViT/ViT_FASHIONMNIST.ipynb`
- `models/ViT/ViT_MNIST.ipynb`

#### **Hybrid ViT-SNN Model:**
- `models/ViTSNN/Hybrid_ViT_SNN_CIFAR10.ipynb`

Once inside a notebook, **run all cells** to train and evaluate the model.

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
| CNN    | CIFAR-10      | 91.08%   | 2172.99 sec  | 12.42 MB  |
| ViT    | CIFAR-10      | 93.45%   | 8424.16 sec | 327.38 MB  |
| ViTSNN | CIFAR-10      | 73.38%   | 12267.26 sec| ---   |


---

## 📜 Project Documentation
The full project report, including the methodology, results, and conclusion, is available in the `report/` directory.

---

## 🤝 Contributing
Contributions are welcome! If you'd like to improve the repository, feel free to fork it, make changes, and submit a pull request.

---



