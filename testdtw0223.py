from numpy import array, zeros, argmin, inf, equal, ndim
import numpy as np
from scipy.spatial.distance import cdist
import time

def dtw(x, y):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.
    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    # assert len(x)
    # assert len(y)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    # D1 = D0[1:, 1:] # view
    # t0 = time.time()
    for i in range(r):
        for j in range(c):
            D0[i+1, j+1] = abs(x[i]-y[j])
    # print time.time()-t0
    # C = D1.copy()
    for i in range(1,r+1):
        for j in range(1,c+1):
            D0[i, j] += min(D0[i-1, j-1], D0[i-1, j], D0[i, j-1])
    # print time.time() - t0
    # if len(x)==1:
    #     path = zeros(len(y)), range(len(y))
    # elif len(y) == 1:
    #     path = range(len(x)), zeros(len(x))
    # else:
    #     path = 0
    return D0[-1, -1] / (sum(D0.shape)-2)

def fastdtw(x, y, dist):
    """
    Computes Dynamic Time Warping (DTW) of two sequences in a faster way.
    Instead of iterating through each element and calculating each distance,
    this uses the cdist function from scipy (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)
    :param array x: N1*M array
    :param array y: N2*M array
    :param string or func dist: distance parameter for cdist. When string is given, cdist uses optimized functions for the distance metrics.
    If a string is passed, the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    if ndim(x)==1:
        x = x.reshape(-1,1)
    if ndim(y)==1:
        y = y.reshape(-1,1)
    r, c = len(x), len(y)
    D0 = zeros((r + 1, c + 1))
    D0[0, 1:] = inf
    D0[1:, 0] = inf
    D1 = D0[1:, 1:]
    t0 = time.time()
    D0[1:,1:] = cdist(x,y,dist)
    print time.time() -t0
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += min(D0[i, j], D0[i, j+1], D0[i+1, j])
    if len(x)==1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = 0
    return D1[-1, -1] / sum(D1.shape), C, D1, path

# x = np.array(range(0,243,1))
# y = np.array(range(2,245,1))
#
# t0 = time.time()
# dist = dtw(x,y)
# t1 = time.time() - t0
#
#
# print t1
#
# print 'end'