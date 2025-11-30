# deep-learning-from-scratch-
This repository contains a collection of deep learning models and training pipelines implemented entirely in Julia, without relying on high-level frameworks such as PyTorch or TensorFlow.

The goal of this project is to develop a fundamental understanding of the mathematical and algorithmic principles behind modern neural networks — including forward and backward passes, gradient computation, optimization, convolution operations, and autoencoding — by building every component manually.

This code was developed as a personal study project to deepen my conceptual understanding of deep learning and to ensure I can reason about models at the level required for research in theoretical machine learning.

I named this little library fran_torch because my name is Francisco :) it is a minimalistic / homemade / educational PyTorch-style version, but implemented entirely by hand, for studying and deeply understanding the fundamentals (deep learning for me, in this case, not for the model haha).

## **Implemented Features**
### **Core Components** 
- Fully-connected layers
- Convolutional layers 
- Transposed convolutional layers
- Maxpooling layers
- Softmax layers
- Activation functions (ReLU, Sigmoid, Tanh)

### **Models** 
- Multilayer Perceptron (MLP) from scratch
- Convolutional Neural Networks (CNNs)
- Autoencoders

### **Training and Optimization**
- Mini-batch gradient descent
- SGD and ADAM for optimization
- Weight initialization schemes

## **Key Learning Outcomes**
- Explicit implementation of forward and backward passes
- Understanding of gradient flow
- Insights into optimization dynamics
- Hands-on experience with low-level operations in a neural network
- Ability to reason about architecture behavior without framework abstractions


## **How to use it**

To use fran_torch, you just need to define a model as a list of structs, where each element of the list is a layer of the network. In fran_torch, there are only 6 types of layers:
conv2d (forward of a 2D convolution layer)
maxpooling (max pooling layer)
flatten (converts a 2D signal, with — possibly — multiple channels, into a vector)
dense (fully connected layer)
conv2d_transpose (transposed convolution layer)
softmax (softmax layer)


To create each of these layers, you just define a struct and pass the attributes of the layer as parameters. For example, for a 2D convolution I need to define the kernel size, the number of kernels, the stride and the activation function. For a dense layer I need to define the number of output neurons and the activation function… well, easier showing than explaining… here is the definition of a small model inspired by LeNet5:

model = network([conv2d(k=5, channels=6, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 conv2d(k=3, channels=16, stride=1, activation="tanh"),
                 maxpooling(k=2, stride=2),
                 flatten(),
                 dense(n_out=120, activation="tanh"),
                 dense(n_out=84, activation="tanh"),
                 dense(n_out=10, activation="tanh"),
                 softmax() 
], 0, 0)

model = init_weights(model, m_in, n_in);


see how easy it is? haha
To perform one training epoch, just use the function train_nn, and to perform full training just put it inside a loop. Here, you can also choose the optimizer to use: Adam ou SGD.
To evaluate the model on a given dataset, just use the function evaluate_nn.

In the repository, I will also include some examples where I use fran_torch for some simple tasks: training LeNet5 with 10,000 MNIST samples (and evaluating on the test set), training another convolutional network for FMNIST classification also with a 10,000-sample training subset, and training a convolutional autoencoder.
