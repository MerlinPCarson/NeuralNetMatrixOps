#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_relu import relu, relu_grad
from q2_gradcheck import gradcheck_naive


def CE(y, y_hat):
    return(-np.sum(y * np.log(y_hat)) / len(y))


def forward_backward_prop(data, labels, params, dimensions, activation='sigmoid'):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H1, H2, Dy = (dimensions[0], dimensions[1], dimensions[2], dimensions[3])

    W1 = np.reshape(params[ofs:ofs + Dx * H1], (Dx, H1))
    ofs += Dx * H1
    b1 = np.reshape(params[ofs:ofs + H1], (1, H1))
    ofs += H1
    W2 = np.reshape(params[ofs:ofs + H1 * H2], (H1, H2))
    ofs += H1 * H2 
    b2 = np.reshape(params[ofs:ofs + H2], (1, H2))
    ofs += H2
    W3 = np.reshape(params[ofs:ofs + H2 * Dy], (H2, Dy))
    ofs += H2 * Dy
    b3 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    num_examples = data.shape[0]

    # FOWARD PASS

    # calculate signal at 1st hidden layer 
    z1 = data.dot(W1) + b1

    # calculate ouput of 1st hidden layer 
    if activation == 'sigmoid':
        a1 = sigmoid(z1)
    elif activation == 'relu':
        a1 = relu(z1)

    # calculate signal at 2nd hidden layer layer
    z2 = a1.dot(W2) + b2

    # calculate ouput of 2nd hidden layer 
    if activation == 'sigmoid':
        a2 = sigmoid(z2)
    elif activation == 'relu':
        a2 = relu(z2)

    # calculate signal at output layer
    z3 = a2.dot(W3) + b3
    a3 = softmax(z3)

    # error on from forward pass
    cost = CE(labels, a3)

    # BACKWARD PASS

    # gradient of weights at output layer
    delta3 = a3 - labels

    gradW3 = a2.T.dot(delta3)/num_examples
    gradb3 = np.sum(delta3, axis=0, keepdims=True)/num_examples

    # gradient of weights at 2nd hidden layer
    if activation == 'sigmoid':
        delta2 = delta3.dot(W3.T) * sigmoid_grad(a2)
    elif activation == 'relu':
        delta2 = delta3.dot(W3.T) * relu_grad(a2)

    gradW2 = a1.T.dot(delta2)/num_examples
    gradb2 = np.sum(delta2, axis=0, keepdims=True)/num_examples

    # gradient of weights at 1st hidden layer
    if activation == 'sigmoid':
        delta1 = delta2.dot(W2.T) * sigmoid_grad(a1)
    elif activation == 'relu':
        delta1 = delta2.dot(W2.T) * relu_grad(a1)

    gradW1 = data.T.dot(delta1)/num_examples
    gradb1 = np.sum(delta1, axis=0, keepdims=True)/num_examples

    assert W1.shape == gradW1.shape
    assert W2.shape == gradW2.shape
    assert W3.shape == gradW3.shape
    assert b1.shape == gradb1.shape
    assert b2.shape == gradb2.shape
    assert b3.shape == gradb3.shape

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten(),
                           gradW3.flatten(), gradb3.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[3]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[3]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * (dimensions[2]) + (dimensions[2] + 1) * dimensions[3], )

    gradcheck_naive(lambda params: forward_backward_prop(
        data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
