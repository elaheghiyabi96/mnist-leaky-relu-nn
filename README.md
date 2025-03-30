# mnist-leaky-relu-nn
A simple neural network implemented from scratch using NumPy to classify MNIST handwritten digits. Uses Leaky ReLU activation and softmax for output.
MNIST Classification Using a Neural Network from Scratch
This project implements a simple neural network from scratch using NumPy to classify the MNIST handwritten digits dataset. The model consists of two fully connected layers and uses the Leaky ReLU activation function for improved gradient flow.

Implementation Details:
Dataset: The MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits (0-9).

Preprocessing:

Pixel values are normalized to the range [0,1] by dividing by 255.

The images are flattened from 28×28 into a 784-dimensional vector for input to the network.

Labels are one-hot encoded for multi-class classification.

Network Architecture:

Input Layer: 784 neurons (flattened 28×28 images)

Hidden Layer: 128 neurons with Leaky ReLU activation (α = 0.004)

Output Layer: 10 neurons with Softmax activation for multi-class classification

Weight Initialization: He initialization (np.sqrt(2/n)) is used to improve convergence.

Loss Function: Categorical Cross-Entropy with a small numerical stability term (1e-8).

Optimization: Gradient Descent with a learning rate of 0.3.

Training: The model was trained for 400 epochs.

Performance:
After training, the model achieved an accuracy of 94.63% on the MNIST dataset.

Possible Improvements:
Implement Batch Normalization to improve stability.

Use an adaptive learning rate such as Adam or RMSprop instead of a fixed learning rate.

Increase the number of hidden layers for better feature extraction.

Train the model with Dropout to prevent overfitting.
#NeuralNetwork 
#DeepLearning 
#MachineLearning 
#NumPy 
#MNIST 
#AI 
#HandwrittenDigitRecognition 
#DataScience 
#Python 
#LeakyReLU
import numpy as np  # Import NumPy for numerical operations
from tensorflow.keras.datasets import mnist  # Import MNIST dataset from TensorFlow
from tensorflow.keras.utils import to_categorical  # Import utility for one-hot encoding

# Load MNIST dataset (handwritten digits 0-9)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1] for better convergence
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape input images from (28, 28) to (784,) vectors for the neural network
x_train = x_train.reshape(-1, 28*28)
x_test = x_test.reshape(-1, 28*28)

# Convert class labels (0-9) into one-hot encoded vectors
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Initialize weights and biases using He initialization for better training stability
w1 = np.random.randn(28*28, 128) * np.sqrt(2 / (28*28))  # First layer weights
b1 = np.zeros((1, 128))  # First layer biases
w2 = np.random.randn(128, 10) * np.sqrt(2 / 128)  # Second layer weights
b2 = np.zeros((1, 10))  # Second layer biases

# Define hyperparameters for training
epochs = 400  # Number of training iterations
learning_rate = 0.3  # Step size for weight updates

# Training loop
for epoch in range(epochs):
    # Forward pass: Compute activations for the first hidden layer
    z1 = x_train @ w1 + b1  # Linear transformation
    a1 = np.maximum(0.004 * z1, z1)  # Leaky ReLU activation function

    # Compute activations for the output layer
    z2 = a1 @ w2 + b2  # Linear transformation for output layer
    z2 -= np.max(z2, axis=1, keepdims=True)  # Prevent numerical overflow in softmax
    y_pred = np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)  # Softmax activation

    # Compute categorical cross-entropy loss (1e-8 added for numerical stability)
    loss = -np.sum(y_train * np.log(y_pred + 1e-8)) / y_train.shape[0]

    # Backpropagation: Compute gradients for the output layer
    dz2 = y_pred - y_train  # Gradient of loss w.r.t. output layer activation
    dw2 = np.dot(a1.T, dz2) / y_train.shape[0]  # Gradient w.r.t. second layer weights
    db2 = np.sum(dz2, axis=0, keepdims=True) / y_train.shape[0]  # Gradient w.r.t. second layer biases

    # Compute gradients for the first hidden layer
    dz1 = np.dot(dz2, w2.T) * ((z1 > 0) + 0.004 * (z1 <= 0))  # Leaky ReLU derivative
    dw1 = np.dot(x_train.T, dz1) / x_train.shape[0]  # Gradient w.r.t. first layer weights
    db1 = np.sum(dz1, axis=0, keepdims=True) / x_train.shape[0]  # Gradient w.r.t. first layer biases

    # Update parameters using gradient descent
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2

    # Print training progress every 10 epochs
    if epoch % 10 == 0:
        y_pred_labels = np.argmax(y_pred, axis=1)  # Convert softmax probabilities to class labels
        y_true_labels = np.argmax(y_train, axis=1)  # Extract true labels
        accuracy = np.mean(y_pred_labels == y_true_labels)  # Compute training accuracy
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

  
