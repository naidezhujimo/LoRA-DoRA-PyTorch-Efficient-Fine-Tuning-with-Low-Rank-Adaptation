# LoRA and DoRA Implementation in PyTorch

This repository contains an implementation of **Low-Rank Adaptation (LoRA)** and **Directional Low-Rank Adaptation (DoRA)** for neural networks using PyTorch. These techniques are useful for fine-tuning large models efficiently by introducing low-rank updates to the model's weights.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Overview](#code-overview)
5. [Examples](#examples)
6. [License](#license)

---

## Introduction

**LoRA (Low-Rank Adaptation)** is a technique that introduces low-rank updates to the weights of a neural network, allowing for efficient fine-tuning of large models. Instead of updating the full weight matrix, LoRA decomposes the update into two smaller matrices, reducing the number of trainable parameters.

**DoRA (Directional Low-Rank Adaptation)** extends LoRA by incorporating directional information into the weight updates. This helps in better aligning the updates with the original weight matrix, potentially improving fine-tuning performance.

This repository provides a PyTorch implementation of both LoRA and DoRA, along with examples of how to integrate them into a neural network.

---

## Installation

To use this code, you need to have Python and PyTorch installed. You can install the required dependencies using the following command:

```bash
pip install torch
```

Clone this repository to your local machine:

```bash
git clone https://github.com/your-username/lora-dora-pytorch.git
cd lora-dora-pytorch
```

---

## Usage

### LoRA Implementation

The `LoRALayer` class implements the low-rank adaptation for a given linear layer. You can wrap any `nn.Linear` layer with the `LinearWithLoRA` class to add LoRA updates.

Example:
```python
layer = nn.Linear(10, 2)
x = torch.randn((1, 10))
layer_lora = LinearWithLoRA(layer, rank=2, alpha=4)
print('Original output:', layer(x))
print('LoRA output:', layer_lora(x))
```

### DoRA Implementation

The `LinearWithDoRAMerged` class implements the Directional Low-Rank Adaptation (DoRA) by merging the LoRA updates with the original weights and applying directional scaling.

Example:
```python
layer = nn.Linear(10, 2)
x = torch.randn((1, 10))
layer_dora = LinearWithDoRAMerged(layer, rank=2, alpha=4)
print('Original output:', layer(x))
print('DoRA output:', layer_dora(x))
```

### Freezing Linear Layers

You can freeze the original linear layers while training only the LoRA or DoRA parameters using the `freeze_linear_layers` function.

Example:
```python
model = MLP(784, 128, 256, 10)
freeze_linear_layers(model)
for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")
```

---

## Code Overview

- **LoRALayer**: Implements the low-rank adaptation using two matrices `A` and `B`.
- **LinearWithLoRA**: Wraps a linear layer with LoRA updates.
- **LinearWithDoRAMerged**: Extends LoRA by incorporating directional scaling (DoRA).
- **MLP**: A simple multi-layer perceptron (MLP) model for demonstration.
- **freeze_linear_layers**: Utility function to freeze the original linear layers.

---

## Examples

### Adding LoRA to an MLP

```python
model = MLP(784, 128, 256, 10)
model.layers[0] = LinearWithLoRA(model.layers[0], rank=4, alpha=8)
model.layers[2] = LinearWithLoRA(model.layers[2], rank=4, alpha=8)
model.layers[4] = LinearWithLoRA(model.layers[4], rank=4, alpha=8)
print(model)
```

### Training with LoRA

You can train the model while keeping the original weights frozen and only updating the LoRA parameters.

---
Feel free to contribute to this repository by opening issues or submitting pull requests!

---

You can customize this `README.md` further to include additional details, such as citation information, references, or acknowledgments.
