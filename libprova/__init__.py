import numpy as np
import libprova.kmeans
import libprova.VIP

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

def entropy(si,s):
    """
    Calculates the single entropy of a parameter
    si: samples of a "i" out
    s: all the samples count
    """
    if si==0:
        return 0
    else:
        return -(si/s)*np.log2(si/s)

def overall_entropy(si_s):
    """
    Calculates the overall entropy
    si_s: np.array with all the samples count
    """
    a = si_s.shape[0]
    sum = 0
    s = np.sum(si_s)
    for i in range(0,a):
        sum += entropy(si_s[i],s)
    return sum