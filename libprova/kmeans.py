import numpy as np

def calc_distance(x1,x2):
    """
    Calculates the distance between two points in R2
    using the euclidean distance
    
    x1,x2: np.array of (2,) shape
    """
    return np.sqrt((x1[0]-x2[0])**2+(x1[1]-x2[1])**2)

def get_closest_centroid(x,centroids):
    """
    Gets the closest centroid from a sample x
    
    x: np.array(2,)
    centroids: np.array(3,2)
    """
    dist = calc_distance(x,centroids[0,:])
    result = 0
    if dist>calc_distance(x,centroids[1,:]):
        dist = calc_distance(x,centroids[1,:])
        result = 1
    elif dist>calc_distance(x,centroids[2,:]):
        dist = calc_distance(x,centroids[2,:])
        result = 2
    return result

def classify_data(X,centroids):
    """
    Returns vector of the closest centroid (between 3) from a set of samples
    """
    size = X.shape[0]
    Y = np.zeros([size])
    for i in range(0,size):
        Y[i] = get_closest_centroid(X[i,:],centroids)
    return Y

def get_new_centroids(X,Y):
    """
    Returns new 3 centroids given X samples and Y categories (0-1-2)
    """
    unique, counts = np.unique(Y, return_counts=True)
    size = X.shape[0]
    new_centroids = np.zeros([3,2])
    for i in range(0,size):
        if Y[i]==0:
            new_centroids[0,:] = new_centroids[0,:] + X[i,:]
        if Y[i]==1:
            new_centroids[1,:] = new_centroids[1,:] + X[i,:]
        if Y[i]==2:
            new_centroids[2,:] = new_centroids[2,:] + X[i,:]
    new_centroids[0,:]=new_centroids[0,:]/counts[0]
    new_centroids[1,:]=new_centroids[1,:]/counts[1]
    new_centroids[2,:]=new_centroids[2,:]/counts[2]
    return new_centroids