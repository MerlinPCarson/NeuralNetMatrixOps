#!/usr/bin/env python

import numpy as np


def relu(x):
    """
    Compute the relu function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- relu(x)
    """

    s = np.maximum(0,x)

    return s


def relu_grad(s):
    """
    Compute the gradient for the relu function here. Note that
    for this implementation, the input s should be the relu
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ds = np.heaviside(s, 0)

    return ds


def test_relu_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = relu(x)
    g = relu_grad(f)
    print(f)
    f_ans = np.array([
        [1, 2],
        [0, 0]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print(g)
    g_ans = np.array([
        [1, 1],
        [0, 0]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print("You should verify these results by hand!\n")


def test_relu():
    """
    Use this space to test your relu implementation by running:
        python q2_relu.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    ####################
    # your answer here
    ####################
    print("Running your tests...")

    # END YOUR CODE


if __name__ == "__main__":
    test_relu_basic()
    test_relu()
