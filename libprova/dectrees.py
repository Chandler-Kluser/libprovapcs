import numpy as np

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

def parameter_entropy_sum(si_s):
    """
    Calculates the overall entropy SUM of a parameter
    si_s: np.array with all the samples count
    """
    a = si_s.shape[0]
    sum = 0
    s = np.sum(si_s)
    for i in range(0,a):
        sum += entropy(si_s[i],s)
    return sum

def overall_entropy(s_list):
    """
    Calculates the overall entropy
    s_list: list of outputs lists of observations counts of all parameters
    """
    w2 = []
    w1 = []
    for i in s_list:
        i = np.array(i)

        w1.append(parameter_entropy_sum(i))
        w2.append(np.sum(i))
    w1 = np.array(w1)
    w2 = np.array(w2)
    w2 = w2/np.sum(w2)
    return np.dot(w1,w2)