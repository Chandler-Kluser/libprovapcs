import numpy as np

def sig(x):
    """
    Logistic Function
    Returns sigmoid of x
    """
    return 1/(1+np.exp(-x))

def wsum(w,x):
    """
    Returns the output of a neuron
    w: weight vector
    x: features
    """
    return np.dot(w,x)

def relu(x):
    """
    Retorns ReLu function:
    x, if x>0
    0 otherwise
    """
    if x>0:
        return x
    else:
        return 0