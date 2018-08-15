# from numba import cuda, float32
# import numpy as np
#
# # Controls threads per block and shared memory usage.
# # The computation will be done on blocks of TPBxTPB elements.
# TPB = 16
#
# # @cuda.jit
# # def fast_matmul(A, B, C):
# #     # Define an array in the shared memory
# #     # The size and type of the arrays must be known at compile time
# #     sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
# #     sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
# #
# #     x, y = cuda.grid(2)
# #
# #     tx = cuda.threadIdx.x
# #     ty = cuda.threadIdx.y
# #     bpg = cuda.gridDim.x    # blocks per grid
# #
# #     if x >= C.shape[0] and y >= C.shape[1]:
# #         # Quit if (x, y) is outside of valid C boundary
# #         return
# #
# #     # Each thread computes one element in the result matrix.
# #     # The dot product is chunked into dot products of TPB-long vectors.
# #     tmp = 0.
# #     for i in range(bpg):
# #         # Preload data into shared memory
# #         sA[tx, ty] = A[x, ty + i * TPB]
# #         sB[tx, ty] = B[tx + i * TPB, y]
# #
# #         # Wait until all threads finish preloading
# #         cuda.syncthreads()
# #
# #         # Computes partial product on the shared memory
# #         for j in range(TPB):
# #             tmp += sA[tx, j] * sB[j, ty]
# #
# #         # Wait until all threads finish computing
# #         cuda.syncthreads()
# #
# #     C[x, y] = tmp
# #
# # AA = np.array([[1,2,3],[4,5,6]])
# # BB = np.array([[1,2],[3,4],[5,6]])
# # CC = np.zeros((2,2))
# #
# # fast_matmul(AA,BB,CC)
# #
# # print CC

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
# @jit
# def mandel(x, y, max_iters):
#   """
#     Given the real and imaginary parts of a complex number,
#     determine if it is a candidate for membership in the Mandelbrot
#     set given a fixed number of iterations.
#   """
#   c = complex(x, y)
#   z = 0.0j
#   for i in range(max_iters):
#     z = z*z + c
#     if (z.real*z.real + z.imag*z.imag) >= 4:
#       return i
#
#   return max_iters
#
# mandel_gpu = cuda.jit(device=True)(mandel)

# @cuda.jit
# def cuda_dtw(A,B,result):
#
#   #allocate A,B and dist matrix C
#   sA = cuda.shared.array(shape=(96),dtype=float32)
#   sB = cuda.shared.array(shape=(96),dtype=float32)
#   sC = cuda.shared.array(shape=(97,97),dtype=float32)
#   tempMin= cuda.shared.array(shape=(1),dtype=float32)
#
#   #get A,B from global memory to shared sA and sB
#   x, y = cuda.grid(2)
#
#   tx = cuda.threadIdx.x
#   ty = cuda.threadIdx.y
#
#   ThreadID = tx + cuda.blockDim.x * ty
#   if (ThreadID < 96):
#     sA[ThreadID] = A[cuda.blockIdx.x,ThreadID]
#     sC[0,ThreadID+1]=10000000000
#
#   if ((ThreadID > 95) and (ThreadID < 192)):
#     tempIndex = ThreadID -96
#     sB[tempIndex] = B[cuda.blockIdx.x,tempIndex]
#     sC[tempIndex+1,0] = 10000000000
#
#   cuda.syncthreads()
#
#   x1 = 3*tx
#   x2 = x1+1
#   x3 = x2+1
#   x4 = x3+1
#
#   y1 = 3*ty
#   y2 = y1+1
#   y3 = y2+1
#   y4 = y3+1
#
#   sC[x2,y2]=abs(sA[x1]-sB[y1])
#   sC[x2,y3]=abs(sA[x1]-sB[y2])
#   sC[x2,y4]=abs(sA[x1]-sB[y3])
#
#   sC[x3,y2]=abs(sA[x2]-sB[y1])
#   sC[x3,y3]=abs(sA[x2]-sB[y2])
#   sC[x3,y4]=abs(sA[x2]-sB[y3])
#
#   sC[x4,y2]=abs(sA[x3]-sB[y1])
#   sC[x4,y3]=abs(sA[x3]-sB[y2])
#   sC[x4,y4]=abs(sA[x3]-sB[y3])
#
#   cuda.syncthreads()
#
#
#
#
#   if (tx==0) and (ty==0) :
#     # for i in range(97):
#     #   for j in range(97):
#     #     result[cuda.blockIdx.x,i,j]=sC[i,j]
#     sC[0,0]=0.0
#     for i in range(1,97):
#       for j in range(1,97):
#         tempMin[0] = sC[i-1,j-1]
#         cuda.atomic.min(tempMin,0,sC[i,j-1])
#         cuda.atomic.min(tempMin,0,sC[i-1,j])
#         sC[i,j] +=tempMin[0]
#
#     result[cuda.blockIdx.x] = sC[96,96]/192.0
    #result[cuda.blockIdx.x] = cuda.blockIdx.x
  # startX, startY = cuda.grid(2)
  # # gridX = cuda.gridDim.x * cuda.blockDim.x;
  # gridY = cuda.gridDim.y * cuda.blockDim.y;
  # index = startY + gridY * startX +1;
  # index_x = startX + (startX > startY)*(30 + 1 - 2*startX) +1;
  # index_y = startY + (startX > startY)*(30 + 1 - 2*startY) +1;
  #
  #
  #
  # if index < 436 :
  #   D1 = np.zeros((243,243))
  #   for i in range(243):
  #     for j in range(243):
  #       D1[i, j] = abs(image[index_x][i]-image[index_y][j])
  #   for i in range(1,243):
  #     for j in range(1,243):
  #       D1[i, j] += min(D1[i-1,j-1], D1[i-1,j], D1[i,j-1])
  #
  #   result[index] = D1[-1,-1]

  # for x in range(startX, width, gridX):
  #   real = min_x + x * pixel_size_ximage
  #   for y in range(startY, height, gridY):
  #     imag = min_y + y * pixel_size_y
  #     image[y, x] = mandel_gpu(real, imag, iters)
  # image[startX][startY]=cuda.blockDim.y;
# @cuda.jit
# def cuda_dtw(A,B,I,Result):
#
#   sA = cuda.shared.array(shape=(96),dtype=float32)
#   sB = cuda.shared.array(shape=(96),dtype=float32)
#   sC = cuda.shared.array(shape=(97,97),dtype=float32)
#   sI = cuda.shared.array(shape=(3),dtype=int32)
#   tempMin= cuda.shared.array(shape=(1),dtype=float32)
#
#   tx = cuda.threadIdx.x
#   ty = cuda.threadIdx.y
#
#
#   ThreadID = tx + cuda.blockDim.x * ty
#
#   if ThreadID ==0 :
#     sI[cuda.blockIdx.x]=I[cuda.blockIdx.x]
#   cuda.syncthreads()
#
#
#   if (ThreadID < 96):
#     sA[ThreadID]=A[sI[cuda.blockIdx.x]+cuda.blockIdx.x,ThreadID]
#     sC[0,ThreadID+1]=10000000000
#
#   if ((ThreadID > 95) and (ThreadID < 192)):
#     tempIndex = ThreadID -96
#     sB[tempIndex] = B[sI[cuda.blockIdx.x]+cuda.blockIdx.x,tempIndex]
#     sC[tempIndex+1,0] = 10000000000
#
#
#   cuda.syncthreads()
#
#   # x1 = 3*tx
#   # x2 = x1+1
#   # x3 = x2+1
#   # x4 = x3+1
#   #
#   # y1 = 3*ty
#   # y2 = y1+1
#   # y3 = y2+1
#   # y4 = y3+1
#   #
#   # sC[x2,y2]=abs(sA[x1]-sB[y1])
#   # sC[x2,y3]=abs(sA[x1]-sB[y2])
#   # sC[x2,y4]=abs(sA[x1]-sB[y3])
#   #
#   # sC[x3,y2]=abs(sA[x2]-sB[y1])
#   # sC[x3,y3]=abs(sA[x2]-sB[y2])
#   # sC[x3,y4]=abs(sA[x2]-sB[y3])
#   #
#   # sC[x4,y2]=abs(sA[x3]-sB[y1])
#   # sC[x4,y3]=abs(sA[x3]-sB[y2])
#   # sC[x4,y4]=abs(sA[x3]-sB[y3])
#   for i in range(3*tx,3*tx+3):
#     for j in range(3*ty,3*ty+3):
#       sC[i+1,j+1]=abs(sA[i]-sB[j])
#   cuda.syncthreads()
#
#
#
#
#   if (tx==0) and (ty==0) :
#     sC[0,0]=0.0
#     for i in range(1,97):
#       for j in range(1,97):
#         tempMin[0] = sC[i-1,j-1]
#         cuda.atomic.min(tempMin,0,sC[i,j-1])
#         cuda.atomic.min(tempMin,0,sC[i-1,j])
#         sC[i,j] +=tempMin[0]
#     Result[sI[cuda.blockIdx.x]+cuda.blockIdx.x]=sC[96,96]/192.0
#     I[cuda.blockIdx.x] +=3



    # if cuda.blockIdx.x ==0:
    #    d_result[d_i[0] - 1, d_j[0] - 3] = sC[96, 96] / 192.0
    # elif cuda.blockIdx.x==1:
    #    d_result[d_i[0] - 1, d_j[0] - 2] = sC[96, 96] / 192.0
    # else :
    #    d_result[d_i[0] - 1, d_j[0] - 1] = sC[96, 96] / 192.0

# @cuda.jit
# def cuda_dtw(A,B,I,D):
#
#   sA = cuda.shared.array(shape=(96),dtype=float32)
#   sB = cuda.shared.array(shape=(96),dtype=float32)
# #  sC = cuda.shared.array(shape=(97,97),dtype=float32)
#   sI = cuda.shared.array(shape=(3),dtype=int32)
#   #
#   tx = cuda.threadIdx.x
#   ty = cuda.threadIdx.y
#
#
#   ThreadID = tx + cuda.blockDim.x * ty
#
#   if ThreadID ==0 :
#     sI[cuda.blockIdx.x]=I[cuda.blockIdx.x]
#     I[cuda.blockIdx.x] += 3
#   cuda.syncthreads()
#
#
#   if (ThreadID < 96):
#     sA[ThreadID]=A[sI[cuda.blockIdx.x]+cuda.blockIdx.x,ThreadID]
#     D[sI[cuda.blockIdx.x]+cuda.blockIdx.x,0,ThreadID+1]=10000000000
#
#   if ((ThreadID > 95) and (ThreadID < 192)):
#     tempIndex = ThreadID -96
#     sB[tempIndex] = B[sI[cuda.blockIdx.x]+cuda.blockIdx.x,tempIndex]
#     D[sI[cuda.blockIdx.x]+cuda.blockIdx.x,tempIndex+1,0] = 10000000000
#
#
#   cuda.syncthreads()
#
#   for i in range(3*tx,3*tx+3):
#     for j in range(3*ty,3*ty+3):
#       D[sI[cuda.blockIdx.x]+cuda.blockIdx.x,i+1,j+1]=abs(sA[i]-sB[j])
#
#   cuda.syncthreads()
#
#   # if (cuda.threadIdx.x==0) and (cuda.threadIdx.y==0) :
#   #   # for i in range(97):
#   #   #   for j in range(97):
#   #   #      D[sI[cuda.blockIdx.x]+cuda.blockIdx.x][i][j] = sC[i][j]
#
#
#
# @cuda.jit
# def compute_min_value(D,Result):
#   indexD = cuda.threadIdx.x+cuda.blockIdx.x*160
#
#
#   D[indexD,0,0]=0.0
#   for i in range(1,97):
#     for j in range(1,97):
#       D[indexD,i,j] +=min(D[indexD,i-1,j-1],D[indexD,i,j-1],D[indexD,i-1,j])
#   Result[indexD]=D[indexD,96,96]/192.0
#
#     # if cuda.blockIdx.x ==0:
#     #    d_result[d_i[0] - 1, d_j[0] - 3] = sC[96, 96] / 192.0
#     # elif cuda.blockIdx.x==1:
#     #    d_result[d_i[0] - 1, d_j[0] - 2] = sC[96, 96] / 192.0
#     # else :
#     #    d_result[d_i[0] - 1, d_j[0] - 1] = sC[96, 96] / 192.0
@cuda.jit
def cuda_dtw(A,B,C,D):

  sA = cuda.shared.array(shape=(88),dtype=float32)
  sB = cuda.shared.array(shape=(88),dtype=float32)

  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y

  ThreadID = tx + cuda.blockDim.x * ty


  if (ThreadID < 88):
    sA[ThreadID]=A[cuda.blockIdx.x,ThreadID]
    D[cuda.blockIdx.x,0,ThreadID+1]=10000000000

  if ((ThreadID > 87) and (ThreadID < 176)):
    tempIndex = ThreadID -88
    sB[tempIndex] = B[cuda.blockIdx.x,tempIndex]
    D[cuda.blockIdx.x,tempIndex+1,0] = 10000000000
  cuda.syncthreads()

  for i in range(12*ty,12*ty+12):
    for j in range(2*tx,2*tx+2):
      D[cuda.blockIdx.x,i+1,j+1]=abs(sA[i]-sB[j])

  cuda.syncthreads()

A=np.random.randn(1,88)
B=np.random.randn(1,88)
C=np.zeros((1,400),dtype=np.float32)
D=np.zeros((1,96,96),dtype=np.float32)
d_D=cuda.to_device(D)
d_A=cuda.to_device(A)
d_B=cuda.to_device(B)
d_C=cuda.to_device(C)
cuda_dtw[(1),(44,8)](d_A,d_A,d_C,d_D)
d_C.copy_to_host(C)
# d_D.copy_to_host(D)
D = d_D.copy_to_host()
print 'OK'