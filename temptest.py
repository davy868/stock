#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
#from pylab import imshow, show
from timeit import default_timer as timer
from numba import cuda
from numba import *
from numpy import array, zeros, argmin, inf, equal, ndim
from find_pairs import *
from testdtw0223 import *
import operator
import csv
import math
from matplotlib import pyplot

g=range(0,5)
print g
g=[i+1 for i in g[0:2]]
print g

Matrixte=np.zeros((1000000,25),dtype=np.float32)


for i in range(len(Matrixte)):
	for j in range(25):
		Matrixte[i][j] = i + j



#plt.plot(Matrixte[1][:])
#plt.plot(Matrixte[2][:])

pyplot.hist(Matrixte[1][:],100,normed=True,histtype='step',cumulative=True)
pyplot.show()

#plt.show()

print "OK"