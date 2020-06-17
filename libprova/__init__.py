import numpy as np
from scipy.stats import norm
import libprova.kmeans
import libprova.VIP
import libprova.dectrees
import libprova.linearregression
import libprova.confusionmatrix

def sig(x):
    """
    Logistic Function, returns sigmoid of x
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

def threshold(x,th=0):
    """
    Returns a step (1-0) given a threshold value
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

def prob_2_z(alpha_2):
    """
    Returns z-value from the standard normal distribution given its alpha_2 area (only one side)

    alpha_2: float
    """
    return norm.ppf(alpha_2)

def wald_test(a,b):
    """
    Returns the W factor of the wald test given the lists of accuracies of necessarilly two classifiers
    a,b : list os both classifiers, must have the same length
    """
    sum_a_minus_b = 0
    sum_2 = 0
    k = len(a)
    for i in range(0,k):
        aux = a[i]-b[i]
        print('----------- i = ',i,' -----------')
        print('a[i]-b[i] = ',aux)
        sum_a_minus_b += aux
    for i in range(0,k):
        aux = a[i]-b[i] - sum_a_minus_b/k
        print('----------- i = ',i,' -----------')
        print('a[i]-b[i] - sum_a_minus_b/k = ',aux)
        sum_2 += aux**2
    W = (   sum_a_minus_b/k  )    /np.sqrt(  sum_2 / (  k*(k-1)  )    )
    print('W = ',W)
    return W

def get_accuracy(err):
    """
    returns a list of the accuracy given the measure errors
    err: list
    """
    acc = []
    for i in err:
        acc.append(1-i)
    return acc