def precision(TP,FP,show=True):
    """
    Returns precision value
    TP : float, true positive samples count
    FP : float, false positive samples count
    """
    precision = TP/(TP+FP)
    if show:
        print('precision =',precision)
    return precision

def recall(TP,FN,show=True):
    """
    Returns recall value
    TP : float, true positive samples count
    FN : float, false negative samples count
    """
    recall = TP/(TP+FN)
    if show:
        print('recall =',recall)
    return recall

def accuracy(TP,TN,FP,FN):
    """
    Returns recall value
    TP : float, true positive samples count
    TN : float, false positive samples count
    FP : float, false positive samples count
    FN : float, false negative samples count
    """
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    print('accuracy =',accuracy)
    return accuracy

def f1(TP,FP,FN):
    """
    Returns f1-score value
    TP : float, true positive samples count
    FP : float, false positive samples count
    FN : float, false negative samples count
    """
    prec = precision(TP,FP,show=False)
    rec = recall(TP,FN,show=False)
    f1 = 2*prec*rec/(prec+rec)
    print('f1 score = ',f1)
    return f1

def showmetrics(TP,TN,FP,FN):
    """
    Returns list of Accuracy, Precision and Recall values in this order
    TP : float, true positive samples count
    TN : float, false positive samples count
    FP : float, false positive samples count
    FN : float, false negative samples count
    """
    acc = accuracy(TP,TN,FP,FN)
    prec = precision(TP,FP)
    rec = recall(TP,FN)
    f1score = f1(TP,FP,FN)
    return [acc,prec,rec,f1score]