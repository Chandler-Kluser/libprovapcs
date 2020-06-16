import numpy as np
import libprova.kmeans
import libprova.VIP
import libprova.dectrees

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

def threshold(x,th):
    """
    Returns a step (1-0) given a treshhold value
    """
    if x>th:
        return 1
    else:
        return 0

def dist(r,q):
    """
    Returns the dist value between two categorical vectors
    r: length of the vectors (for instance [0,0,0] --> r=3)
    q: how many coordinates MATCHES between the two vectors (for instance [1,1,0] and [1,0,0] --> q=2)
    """
    return (r-q)/r