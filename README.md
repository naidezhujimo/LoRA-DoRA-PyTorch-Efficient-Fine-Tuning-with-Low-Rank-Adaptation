# MNIST Classification with LoRA, DoRA, and QLoRA

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) for classifying the MNIST dataset. The project explores different fine-tuning techniques, including LoRA (Low-Rank Adaptation), DoRA (Dynamic Low-Rank Adaptation), and QLoRA (Quantized Low-Rank Adaptation), to improve model performance and efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to compare the performance of different fine-tuning methods on the MNIST dataset. The base model is a simple CNN, and we apply LoRA, DoRA, and QLoRA techniques to fine-tune the model. The results are compared in terms of accuracy and memory usage.

### Techniques

- **LoRA (Low-Rank Adaptation)**: A technique that introduces low-rank matrices to adapt the model's weights, reducing the number of trainable parameters.
- **DoRA (Dynamic Low-Rank Adaptation)**: An extension of LoRA that dynamically adjusts the rank of the adaptation matrices.
- **QLoRA (Quantized Low-Rank Adaptation)**: A quantized version of LoRA that reduces the precision of the weights to save memory and computational resources.

## Installation

To run this project, you need to have Python 3.7 or later installed. Follow these steps to set up the environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mnist-lora-dora-qlora.git
   cd mnist-lora-dora-qlora
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the models and generate the results, run the following command:

```bash
python main.py
```

This script will:

1. Load the MNIST dataset.
2. Train the base CNN model.
3. Apply LoRA, DoRA, and QLoRA techniques and train the respective models.
4. Generate plots comparing the training loss, validation accuracy, and memory usage of the different methods.

### Hyperparameters

You can adjust the hyperparameters in the script to experiment with different settings:

- `BATCH_SIZE`: Batch size for training and testing.
- `learning_rate`: Learning rate for the optimizer.
- `rank`: Rank of the low-rank matrices in LoRA, DoRA, and QLoRA.
- `alpha`: Scaling factor for the low-rank adaptation.
- `num_epochs`: Number of training epochs.

## Results

The script will generate the following plots:

1. **Training Loss Comparison**: Compares the training loss of the base model, LoRA, DoRA, and QLoRA.
2. **Validation Accuracy Comparison**: Compares the validation accuracy of the base model, LoRA, DoRA, and QLoRA.
3. **Base Model: Loss and Accuracy**: Shows the training loss and validation accuracy for the base model.
4. **LoRA: Loss and Accuracy**: Shows the training loss and validation accuracy for the LoRA model.
5. **Comparison of Test Accuracy and Memory Usage**: A bar chart comparing the final test accuracy and memory usage of all methods.

### Example Output

After running the script, you should see output similar to the following:

```
Training base model...
Epoch 1, Loss: 0.1234, Learning Rate: 0.000100, Val Accuracy: 98.50%
...
Early stopping at epoch 25
Total Training Time: 5.23 min
Max memory allocated: 512.34 MB
-------------------------------------
Training LoRA model...
Epoch 1, Loss: 0.1234, Learning Rate: 0.000100, Val Accuracy: 98.50%
...
Early stopping at epoch 25
Total Training Time: 5.23 min
Max memory allocated: 512.34 MB
-------------------------------------
...
Base Model Test Accuracy: 98.50%
LoRA Model Test Accuracy: 98.60%
DoRA Model Test Accuracy: 98.70%
QLoRA Model Test Accuracy: 98.55%
```

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

