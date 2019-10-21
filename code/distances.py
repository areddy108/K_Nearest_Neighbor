import numpy as np 

def euclidean_distances(X, Y):
    if(X.ndim == 1):
        X = np.reshape(X, (1, np.size(X, 0)))

    if(Y.ndim == 1):
        Y = np.reshape(Y, (1, np.size(Y, 0)))

    D = np.zeros((np.size(X,0), np.size(Y,0)))


    for i in range(np.size(X, 0)):
        for j in range(np.size(Y, 0)):
            D[i, j] = np.sqrt(np.sum(np.power(np.subtract(X[i, :], Y[j, :]), 2)))

    return D

    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    #raise NotImplementedError()


def manhattan_distances(X, Y):
    D = np.zeros((np.size(X, 0), np.size(Y, 0)))

    for i in range(np.size(X, 0)):
        for j in range(np.size(Y, 0)):
            D[i, j] = np.sum(np.abs(np.subtract(X[i, :], Y[j, :])))

    return D



    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    #raise NotImplementedError()


def cosine_distances(X, Y):
    D = np.zeros((np.size(X, 0), np.size(Y, 0)))

    for i in range(np.size(X, 0)):
        for j in range(np.size(Y, 0)):
            D[i, j] = 1 - np.dot(X[i, :], Y[j, :]) / (np.linalg.norm(X[i, :]) * np.linalg.norm(Y[j, :]))

    return D

    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    #raise NotImplementedError()