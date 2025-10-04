# Exploration of CNN architectures on CIFAR-10 

This project explores the performance of different Convolutional Neural Network (CNN) architectures on the CIFAR-10 image classification task, implemented entirely from scratch without deep learning libraries such as TensorFlow/Keras/PyTorch.  

We designed a modular framework that implements fundamental neural network components such as convolutional layers, pooling layers, activation functions, softmax, and cross-entropy loss, and then used it to train and compare five CNN variants.  

> Note: The models achieved modest accuracies (~15–28%). The main focus was on **understanding CNN mechanics, backpropagation, and weight updates**, not achieving state-of-the-art results. This is the course project that was built on top of the course assignments.

## Objectives  
- Build a **custom CNN framework** from first principles.  
- Compare **single vs. multi-kernel CNNs** on grayscale and color images.  
- Explore the impact of **deeper CNN architectures**.  
- Understand **training behavior, gradient flow, and feature extraction**.  

## Implemented Architectures  
1. **Single-Kernel Grayscale CNN**  
2. **Multi-Kernel Grayscale CNN**  
3. **Single-Kernel Color CNN**  
4. **Multi-Kernel Color CNN**  
5. **Multi-layer Grayscale CNN**  

Each model is implemented using the framework and trained on CIFAR-10 subsets.  

## Framework Design  
The framework (`framework/`) folder contains reusable building blocks:

- `Layer.py` – Abstract base class for forward/backward/gradient methods  
- `InputLayer.py` – Handles input and optional z-score normalization  
- `ConvolutionalLayer.py` – Standard 2D convolution (grayscale)  
- `Convolutional3DLayer.py` – 3D convolution (color images)  
- `MaxPoolLayer.py` – Max pooling with configurable size and stride  
- `FlatteningLayer.py` – Converts 3D feature maps into vectors  
- `FullyConnectedLayer.py` – Dense layer with weight/bias updates  
- `LogisticSigmoidLayer.py` – Sigmoid activation  
- `SoftmaxLayer.py` – Softmax activation for classification  
- `CrossEntropy.py` – Loss function for training

The framework folder is not shared in the repository as it is part of course assignments.

## Dataset  
- **[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)**: 60,000 images (32×32, RGB, 10 classes) with 50,000 training and 10,000 testing. 
<!-- - Training: 50,000 images | Testing: 10,000 images.  -->
- For this project, a subset of **5,000 training images (500 per class)** + **5,000 test images** was used for faster runs.  
- Grayscale experiments used pre-processed CIFAR-10 with 1 channel.    


## Training Setup  
- **Initialization**: He initialization for weights  
- **Optimization**: Gradient Descent + L2 regularization  
- **Hyperparameters**:  
  - Learning rate: `1e-2` (single/multi-kernel), `1e-3` (multi-layer CNN)  
  - Epochs: 100 (with early stopping if loss change < 1e-10)  
- **Loss function**: Cross-Entropy  
- **Evaluation**: Accuracy and loss curves  


## Repository Structure
```
.  
├── framework/
├── cifar-10-testing-grayscale/ #dataset
├── cifar-10-training-grayscale/ #dataset
├── cifar-10-testing-color/ #dataset
├── cifar-10-training-color/ #dataset
├── Single-Kernel-CNN-grayscale.py
├── Multi-Kernel-CNN-grayscale.py
├── Single-Kernel-CNN-color.py
├── Multi-Kernel-CNN-color.py
├── Multi-Layer-Single-Kernel-CNN-grayscale.py
├── docs/
├── README.md
```
## Architecture Design & Dimension Flow  

Each CNN was implemented using our **custom framework**, and we tracked how the **input dimensions transform at each layer**.  

> CIFAR-10 Input: **32×32×3 (color)** or **32×32×1 (grayscale)**  
>  
> Convolution (no padding, stride=1):  
> \[(H, W) → (H − k + 1, W − k + 1)\]  
>  
> Max Pooling (size=2, stride=2):  
> \[(H, W) → (H/2, W/2)\]

### 1. Single-Kernel Grayscale CNN  
- **Input**: 32×32×1  
- **Conv2D (k=3, 1 kernel)** → 30×30×1  
- **MaxPool (2×2, stride=2)** → 15×15×1  
- **Flatten** → 225 nodes  
- **Fully Connected (225 → 10)**  
- **Softmax + CrossEntropy** → class probabilities 

Smallest CNN, serves as baseline.  

### 2. Multi-Kernel Grayscale CNN  
- **Input**: 32×32×1  
- **Conv2D (k=3, 3 kernels)** → 30×30×3  
- **MaxPool (2×2, stride=2)** → 15×15×3  
- **Flatten** → 675 nodes  
- **Fully Connected (675 → 10)**  
- **Softmax + CrossEntropy** → class probabilities

More kernels allow extraction of **multiple feature maps** (edges, corners, textures).  

### 3. Single-Kernel Color CNN  
- **Input**: 32×32×3  
- **Conv3D (k=3×3×3, 1 kernel)** → 30×30×1  
- **MaxPool (2×2, stride=2)** → 15×15×1  
- **Flatten** → 225 nodes  
- **Fully Connected (225 → 10)**  
- **Softmax + CrossEntropy** → class probabilities 

Uses a 3D kernel across RGB channels, but with **only one kernel**, feature diversity is low.

### 4. Multi-Kernel Color CNN  
- **Input**: 32×32×3  
- **Conv3D (k=3×3×3, 3 kernels)** → 30×30×3  
- **MaxPool (2×2, stride=2)** → 15×15×3  
- **Flatten** → 675 nodes  
- **Fully Connected (675 → 10)**  
- **Softmax + CrossEntropy** → class probabilities 

Best-performing model: multiple 3D kernels capture richer **color + spatial features**.

### 5. Multi-Layer Grayscale CNN  
- **Input**: 32×32×1  
- **Conv2D (k=3, 1 kernel)** → 30×30×1 
- **MaxPool (2×2, stride=2)** → 15×15×1 
- **Conv2D (k=3, 1 kernel)** → 13×13×1 
- **MaxPool (2×2, stride=2)** → 6×6×1 
- **Flatten** → 36 nodes  
- **Fully Connected (36 → 10)**  
- **Softmax + CrossEntropy** → class probabilities 

A deeper CNN, but **limited by dataset size** and lack of augmentation → underfitting. 

## Results

| CNN Architecture | Training Loss | Testing Loss | Training Acc | Testing Acc | Time (s) |
|---|---:|---:|---:|---:|---:|
| Single-Kernel Grayscale | 2.223575 | 2.316045 | 0.2242 | 0.1952 | 5989.12* |
| Multi-Kernel Grayscale | 2.279735 | 2.3441114 | 0.2060 | 0.1816 | 2755.70 |
| Single-Kernel Color | 2.041966 | 2.182492 | 0.2798 | 0.2360 | 8172.74 |
| Multi-Kernel Color  | 1.979867 | 2.050411  | 0.3014 | 0.2812 | 8319.51 |
| Multi-layer Single-Kernel Grayscale | 2.263160 | 2.269833  | 0.1630 | 0.1524 | 3858.47 |

[Loss and Accuracy plots](docs/results.md)

## Key Learnings  
- **Multi-kernel models** captured richer features.  
- **Color inputs** only helped when combined with multiple kernels.  
- **Deeper CNNs** require regularization and augmentation to avoid underfitting.  
- Building the framework clarified **backpropagation, gradient updates, and CNN architecture tradeoffs**.  

## Future Work  
- Try different kernel sizes and pooling strategies 
- Add batch normalization 
- Apply data augmentation for better generalization  
- Extend multi-layer CNNs to color models  





