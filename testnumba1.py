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
def cuda_dtw(A,B,D):

  sA = cuda.shared.array(shape=(96),dtype=float32)
  sB = cuda.shared.array(shape=(96),dtype=float32)

  tx = cuda.threadIdx.x
  ty = cuda.threadIdx.y

  ThreadID = tx + cuda.blockDim.x * ty


  if (ThreadID < 96):
    sA[ThreadID]=A[cuda.blockIdx.x,ThreadID]
    D[cuda.blockIdx.x,0,ThreadID+1]=10000000000

  if ((ThreadID > 95) and (ThreadID < 192)):
    tempIndex = ThreadID -96
    sB[tempIndex] = B[cuda.blockIdx.x,tempIndex]
    D[cuda.blockIdx.x,tempIndex+1,0] = 10000000000
  cuda.syncthreads()

  for i in range(12*ty,12*ty+12):
    for j in range(3*tx,3*tx+3):
      D[cuda.blockIdx.x,i+1,j+1]=abs(sA[i]-sB[j])

  cuda.syncthreads()


@cuda.jit
def compute_min_value(D,Result):
  indexD = cuda.threadIdx.x+cuda.blockIdx.x*128
  D[indexD,0,0]=0.0
  for i in range(1,97):
    for j in range(1,97):
      D[indexD,i,j] +=min(D[indexD,i-1,j-1],D[indexD,i,j-1],D[indexD,i-1,j])
  Result[indexD]=D[indexD,96,96]/192




def get_normalizetion_data(filedir,stock_number): #输入每日分钟数据，返回波动范围在1.02到1.09范围内的归一化分钟数据
	day_prices_in_one_array = get_day_prices_in_one_array(filedir)
	day_prices_in_one_array.astype(np.float32)
	day_price_voltality=np.zeros((1000,244),dtype=np.float32)
	day_price_voltality = choose_high_votality_stock(1.02,1.09,day_prices_in_one_array)

	for i in range(0,len(day_price_voltality)):
		day_price_voltality[i][0:243] = z_score(np.average(day_price_voltality[i][0:243]),np.std(day_price_voltality[i][0:243]),day_price_voltality[i][0:243])
		for j in range(len(stock_number)):
			if stock_number[j] == day_price_voltality[i][243] :
				day_price_voltality[i][243]=j
				break
	#	day_price_voltality[i][0:243] = np.random.normal(0,1,243)
	day_price_voltality.astype(np.float32)
	return day_price_voltality


def dtw_day_price(day_price_voltality):#计算两两股票之间的dtw距离
	lenth_day_price = len(day_price_voltality)
	half_length_day_price = int(math.floor(lenth_day_price/2) + (lenth_day_price%2 != 0)*1)
	day_price_row96 = np.zeros((lenth_day_price, 96), dtype = np.float32)
	A = np.zeros((22898,96),dtype=np.float32)
	allA = np.zeros((500000,96),dtype=np.float32)
	B = np.zeros((22898,96),dtype=np.float32)
	allB = np.zeros((500000,96),dtype=np.float32)
	D = np.zeros((22912,97,97),dtype=np.float32)
	# result=np.zeros((15,30),dtype=np.float32)
	resultAll=np.zeros((22912),dtype=np.float32)
	indexI=np.zeros((3),dtype=np.int32)
	resultForAll = np.zeros((500000),dtype=np.float32)

	for i in range(lenth_day_price):
	  for j in range(96):
	    day_price_row96[i][j]=day_price_voltality[i][int(j*2.53125)]

	blockdim1 = (32,8)
	blockdim2 = (128)
	griddim1 = (22898)
	griddim2 = (179)

	for i in range(1,(half_length_day_price+1)):
		for j in range(1,(lenth_day_price+1)):
			allA[lenth_day_price*i+j-lenth_day_price-1][0:96] = day_price_row96[i+(i>j)*(lenth_day_price+1-2*i)-1][0:96]
			allB[lenth_day_price*i+j-lenth_day_price-1][0:96] = day_price_row96[j+(i>j)*(lenth_day_price+1-2*j)-1][0:96]
		if lenth_day_price%2 != 0 : #当length_day_price不能被2整除时，矩阵最后一行的配对数会重复，所以把重复的配对设置为相同的元素，dtw为0
			allB[(lenth_day_price * half_length_day_price + 1 - lenth_day_price - 1):(lenth_day_price * half_length_day_price + half_length_day_price - lenth_day_price - 1)][:] =allA[(lenth_day_price * half_length_day_price + 1 - lenth_day_price - 1):(lenth_day_price * half_length_day_price + half_length_day_price - lenth_day_price - 1)][:]


	d_D = cuda.to_device(D)
	for i in range(int(math.floor(lenth_day_price*half_length_day_price/22898)+1*(lenth_day_price*half_length_day_price%22898 != 0))):
		A = allA[i*22898 : (i+1)*22898]
		B = allB[i*22898 : (i+1)*22898]
		d_A = cuda.to_device(A)
		d_B = cuda.to_device(B)
		resultAll = 0.0*resultAll
		d_result = cuda.to_device(resultAll)


		cuda_dtw[griddim1, blockdim1](d_A,d_B,d_D)
		# d_D.copy_to_host(D)
		compute_min_value[griddim2, blockdim2](d_D,d_result)
		# d_D.copy_to_host(D)
		d_result.copy_to_host(resultAll)

		resultForAll[i*22898 : (i+1)*22898] = resultAll[0:22898]
	return resultForAll[0:(lenth_day_price*half_length_day_price)]

def get_pairs_from_dtw(resultValid,limit_for_dtw,day_price_voltality):#得到在给定阈值下的配对数据
	lenth_day_price = len(day_price_voltality)
	half_length_day_price = math.floor(lenth_day_price / 2) + (lenth_day_price % 2 != 0) * 1
	pairs_day = np.zeros((len(resultValid), 6), dtype=np.float32)
	x=np.ones_like(resultValid)
	x = np.sort(resultValid)
	sorted_index=np.argsort(resultValid)

	index_t=0
	while (resultValid[sorted_index[index_t+half_length_day_price]]<limit_for_dtw ) and (resultValid[sorted_index[index_t+half_length_day_price]]>0.0 )  :
		j=(sorted_index[index_t+half_length_day_price]+lenth_day_price+1)%lenth_day_price
		i=math.floor((sorted_index[index_t+half_length_day_price]+lenth_day_price+1)/lenth_day_price)
		if j == 0:
			j = lenth_day_price
			i = i-1
		pairs_day[index_t][0] = day_price_voltality[i + (i > j) * (1+lenth_day_price - 2 * i) - 1][243]
		# if j + (i > j) * (1+lenth_day_price - 2 * j) - 1 == 214:
		# 	print "OK"
		pairs_day[index_t][1] = day_price_voltality[j + (i > j) * (1+lenth_day_price - 2 * j) - 1][243]
		index_t +=1
		assert(pairs_day[index_t][0]==pairs_day[index_t][1])

	return pairs_day,index_t
	# with open("pairs0325_index.csv", "w") as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	for element in y:
	# 		writer.writerow([element])
	#
	# with open("pairs0325_value.csv", "w") as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	for element in x:
	# 		writer.writerow([element])
	#
	# with open("pairs0325_value_origin.csv", "w") as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	for element in resultValid:
	# 		writer.writerow([element])
def get_sorted_axle():#获取所有股票的代码，保存在stock_number中
	temp_stock_number = np.zeros((1000*25), dtype=np.float32)
	filedirs = os.listdir('/root/neural-networks-and-deep-learning/csv5minute/month1')
	for i in range(0, len(filedirs)):
		filedirs[i] = '/root/neural-networks-and-deep-learning/csv5minute/month1/' + filedirs[i]
		day_prices_in_one_array = get_day_prices_in_one_array(filedirs[i])
		day_prices_in_one_array.astype(np.float32)
		temp_stock_number[i*1000:(i+1)*1000]=day_prices_in_one_array[:,243]
		# np.append(temp_stock_number,day_prices_in_one_array[:,243])
	stock_number=np.unique(temp_stock_number)
	# stock_number=stock_number-600000.0

	return stock_number[1:]

stock_number=get_sorted_axle()
with open("stock_number_month1.csv", "w") as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(stock_number)


length_of_stock_number=len(stock_number)
stock_pairs_mark=np.zeros((length_of_stock_number*(length_of_stock_number),7),dtype=np.float32)
stock_appears_number=np.zeros((length_of_stock_number),dtype=np.float32)

pairs_day_month = np.zeros((30,500000,6), dtype=np.float32)
filedirs = os.listdir('/root/neural-networks-and-deep-learning/csv5minute/month1')
for i in range(0, len(filedirs)):
    filedirs[i] = '/root/neural-networks-and-deep-learning/csv5minute/month1/' + filedirs[i]
    day_price_voltality = get_normalizetion_data(filedirs[i],stock_number)
    for t in range(len(day_price_voltality)):
	    stock_appears_number[int(day_price_voltality[t][243])]+=1

    resultValid=dtw_day_price(day_price_voltality)
    # pyplot.hist(resultValid, 1000, normed=True, histtype='step', cumulative=True)
    # pyplot.show()

    temp_pairs,vaid_length_of_pairs=get_pairs_from_dtw(resultValid,0.24,day_price_voltality)  #0.24是选取配对的阈值，即dtw小于0.24的，认为配对成功
    pairs_day_month[i][0:len(temp_pairs)][:] = temp_pairs
    for j in range(vaid_length_of_pairs):

	    if pairs_day_month[i][j][0]<pairs_day_month[i][j][1]:
		    stock_pairs_mark[int(-0.5 * pairs_day_month[i][j][0] * pairs_day_month[i][j][0] + (length_of_stock_number - 1.5) *pairs_day_month[i][j][0] + pairs_day_month[i][j][1]),0] =pairs_day_month[i][j][0]
		    stock_pairs_mark[int(-0.5 * pairs_day_month[i][j][0] * pairs_day_month[i][j][0] + (length_of_stock_number - 1.5) *pairs_day_month[i][j][0] + pairs_day_month[i][j][1]),1] = pairs_day_month[i][j][1]
		    stock_pairs_mark[int(-0.5 * pairs_day_month[i][j][0] * pairs_day_month[i][j][0] + (length_of_stock_number - 1.5) *pairs_day_month[i][j][0] + pairs_day_month[i][j][1]),2] +=1.0

	    else:
		    stock_pairs_mark[int(-0.5 * pairs_day_month[i][j][1] * pairs_day_month[i][j][1] + (length_of_stock_number - 1.5) *pairs_day_month[i][j][1] + pairs_day_month[i][j][0]),0] = pairs_day_month[i][j][1]
		    stock_pairs_mark[int(-0.5 * pairs_day_month[i][j][1] * pairs_day_month[i][j][1] + (length_of_stock_number - 1.5) *pairs_day_month[i][j][1] + pairs_day_month[i][j][0]),1] = pairs_day_month[i][j][0]
		    stock_pairs_mark[int(-0.5 * pairs_day_month[i][j][1] * pairs_day_month[i][j][1] + (length_of_stock_number - 1.5) *pairs_day_month[i][j][1] + pairs_day_month[i][j][0]),2] +=1.0




for i in range(len(stock_pairs_mark)):
	if stock_pairs_mark[i,2] >=3 :  #配对次数
		stock_pairs_mark[i,3]=stock_appears_number[stock_pairs_mark[i,0]] #stock No1在一个月内出现波动的次数（波动范围1.02 to 1.09）
		stock_pairs_mark[i, 4] = stock_appears_number[stock_pairs_mark[i, 1]]#stock No2在一个月内出现波动的次数（波动范围1.02 to 1.09）
		stock_pairs_mark[i, 5] = stock_pairs_mark[i,2]/(min(stock_pairs_mark[i,3],stock_pairs_mark[i,4]))#配对的次数占两个股票波动次数的百分比
		stock_pairs_mark[i,6]=stock_pairs_mark[i,5]*(1-pow(0.7,stock_pairs_mark[i,2]))/(1-0.7)#平衡配对次数与配对占比

temp_bingo_value=np.sort(stock_pairs_mark[:,5])
temp_bingo_index=np.argsort(stock_pairs_mark[:,5])
print "OK"
temp_bingo_value=np.sort(stock_pairs_mark[:,6])
temp_bingo_index=np.argsort(stock_pairs_mark[:,6])

with open("stock_pairs_mark_month1.csv", "w") as csvfile:
	writer = csv.writer(csvfile)
	for row  in  stock_pairs_mark[temp_bingo_index[858684:861184],:]:
		writer.writerow(row)


print "OK"