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
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    num_examples = data.shape[0]

    # FOWARD PASS

    # calculate signal at hidden layer 
    z1 = data.dot(W1) + b1

    # calculate ouput of hidden layer 
    if activation == 'sigmoid':
        a1 = sigmoid(z1)
    elif activation == 'relu':
        a1 = relu(z1)

    # calculate signal at output layer
    z2 = a1.dot(W2) + b2
    a2 = softmax(z2)

    # error on from forward pass
    cost = CE(labels, a2)

    # BACKWARD PASS

    # gradient of weights at output layer
    delta2 = a2 - labels

    gradW2 = a1.T.dot(delta2)/num_examples
    gradb2 = np.sum(delta2, axis=0, keepdims=True)/num_examples

    # gradient of weights at hidden layer
    if activation == 'sigmoid':
        delta1 = delta2.dot(W2.T) * sigmoid_grad(a1)
    elif activation == 'relu':
        delta1 = delta2.dot(W2.T) * relu_grad(a1)

    gradW1 = data.T.dot(delta1)/num_examples
    gradb1 = np.sum(delta1, axis=0, keepdims=True)/num_examples

    assert W1.shape == gradW1.shape
    assert W2.shape == gradW2.shape
    assert b1.shape == gradb1.shape
    assert b2.shape == gradb2.shape

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(
        data, labels, params, dimensions, activation='sigmoid'), params)

    gradcheck_naive(lambda params: forward_backward_prop(
        data, labels, params, dimensions, activation='relu'), params)

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
