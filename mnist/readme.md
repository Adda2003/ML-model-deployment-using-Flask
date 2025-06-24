# Training hyperparameters
batch_size    = 64
epochs        = 10
learning_rate = 0.001

# CNN architecture
conv1_out    = 32    # filters in the 1st conv layer
conv2_out    = 64    # filters in the 2nd conv layer
kernel_size1 = 3     # kernel size for conv1
kernel_size2 = 3     # kernel size for conv2
fc1_units    = 128   # neurons in the first FC layer
dropout_p    = 0.25  # dropout probability


folder_name/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
└── predict.py

# Tensor Shapes Through the Network

| Layer              | Operation                       | Output shape    | Explanation                        |
| ------------------ | ------------------------------- | --------------- | ---------------------------------- |
| **Input**          | —                               | (B, 1, 28, 28)  | B=batch, 1 channel, 28×28 pixels   |
| **Conv1**          | Conv2d(1 → 32, k=3, pad=1)      | (B, 32, 28, 28) | 32 filters, preserves spatial size |
| **ReLU + Pool1**   | ReLU → MaxPool2d(2×2, stride=2) | (B, 32, 14, 14) | Halves H/W: 28→14                  |
| **Conv2**          | Conv2d(32 → 64, k=3, pad=1)     | (B, 64, 14, 14) | 64 filters                         |
| **ReLU + Pool2**   | ReLU → MaxPool2d(2×2, stride=2) | (B, 64, 7, 7)   | Halves 14→7                        |
| **Flatten**        | view(–1, 64×7×7)                | (B, 3136)       | Flatten for FC                     |
| **FC1**            | Linear(3136 → 128)              | (B, 128)        | Fully connected                    |
| **ReLU + Dropout** | ReLU → Dropout(0.25)            | (B, 128)        | Dropout for regularization         |
| **FC2**            | Linear(128 → 10)                | (B, 10)         | 10 logits (one per digit class)    |

# Why No Softmax at the End?

The model ends with a linear layer:
x = self.fc2(x)  # shape (B, 10), raw logits

No softmax here:
PyTorch’s nn.CrossEntropyLoss internally applies LogSoftmax and then computes the negative log-likelihood.
This is more stable numerically; adding your own softmax is not needed.

# The Loss: Cross-Entropy Loss

For multi-class classification with logits $z_1, z_2, ..., z_K$ and true class label $j$, the loss is:

\[
\ell = -\log\left( \frac{\exp(z_j)}{\sum_{i=1}^K \exp(z_i)} \right)
   = -z_j + \log\left( \sum_{i=1}^K e^{z_i} \right)
\]

Range: $[0, \infty)$

$\ell = 0$ if the model assigns 100% probability to the correct class ($z_j \gg z_{i \neq j}$)

$\ell \to \infty$ if the model assigns near-zero probability to the correct class.

During training, the average loss over the batch is minimized.
