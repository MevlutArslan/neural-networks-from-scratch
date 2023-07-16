# Neural Network Implementation From Scratch in C

## Description

This project implements a basic neural network library from scratch in C. It provides the core components needed to build and train neural network models, including:

Common layers types like dense (will implement more as I learn more)
Activation functions like ReLU, leaky RELU, softmax, sigmoid, etc.
Loss functions such as categorical cross-entropy, mean squared error.
Optimization algorithms including stochastic gradient descent, ADAGRAD, RMS_PROP, ADAM.
Backpropagation algorithms to calculate gradients.
Data pre-processing and normalization.

The goal of this library is to provide a "playground" for understanding the underlying mathematical concepts and algorithms that drive neural networks and deep learning. By building from the ground up in C, all aspects of the implementation can be learned and customized.

The library can be used to train neural network models on both CPU and GPU via CUDA extensions (coming soon). It provides building blocks that could be used for projects across computer vision, natural language processing, and other problem domains.

While performance and production-readiness is not a priority, this library serves as an educational tool for hands-on learning. Contributions and improvements are welcome!

## Prerequisites

Before you begin, ensure you have met the following requirements:

* You have installed the latest version of `gcc`.
* You have a `<Linux/Mac>` machine. 
* TODO: Provide link to documentation

## Installing 

To install, follow these steps:

1. Install `gnuplot` on your system. The method for this varies depending on your operating system:

   * On Ubuntu, you can use `sudo apt-get install gnuplot`.
   * On macOS, you can use `brew install gnuplot`.

2. Clone the repository:
```bash
git clone <repository_link>
```

3. Navigate to the project directory:
```bash
cd <repository_directory>
```

4. Compile the project:
```bash
make
```