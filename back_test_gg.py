#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
#from pylab import imshow, show
#from timeit import default_timer as timer
#from numba import cuda
#from numba import *
#from numpy import array, zeros, argmin, inf, equal, ndim
#from find_pairs import *
#from testdtw0223 import *
import operator
import csv
import math
import gc
#from base_function_gg  import *

verification_file='/root/neural-networks-and-deep-learning/csv5minute/verification_file/month1501/'

#get_base_G(verification_file)
def back_test_gg(g1,g2,trade_fee):
#######################################
#根据binggo_plus_stock_pairs_mongth1to9.csv中的配对数据，计算14年10月11月12月，15年01月
#相应配对数据的收益，结果保存在stock_pairs_binggo1to9中，并存为binggo_results1yo9.csv文件
#g1:配对一股票的上午收益阈值，g2配对二股票的当天下午和第二天下午收盘时的收益，trade_fee为交易费用
#######################################
	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/base_G_month10.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:20])
	stock_pairs_plus10 = np.zeros((len(temp_stock_pairs_plus), 20), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus10[i][0:20] = map(np.float32, temp_stock_pairs_plus[i][0:20])
	del temp_stock_pairs_plus
	gc.collect()

	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/base_G_month11.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:20])
	stock_pairs_plus11 = np.zeros((len(temp_stock_pairs_plus), 20), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus11[i][0:20] = map(np.float32, temp_stock_pairs_plus[i][0:20])
	del temp_stock_pairs_plus
	gc.collect()

	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/base_G_month12.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:20])
	stock_pairs_plus12 = np.zeros((len(temp_stock_pairs_plus), 20), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus12[i][0:20] = map(np.float32, temp_stock_pairs_plus[i][0:20])
	del temp_stock_pairs_plus
	gc.collect()

	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/base_G_month1501.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:20])
	stock_pairs_plus1501 = np.zeros((len(temp_stock_pairs_plus), 20), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus1501[i][0:20] = map(np.float32, temp_stock_pairs_plus[i][0:20])
	del temp_stock_pairs_plus
	gc.collect()

	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_plus_stock_pairs_mongth1to9.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:20])
	stock_pairs_binggo1to9 = np.zeros((len(temp_stock_pairs_plus), 40), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_binggo1to9[i][0:2] = map(np.float32, temp_stock_pairs_plus[i][0:2])
	del temp_stock_pairs_plus
	gc.collect()

	temp_stock_number = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_number.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_number=item
	stock_number = np.zeros(len(temp_stock_number), float)
	for i in range(len(temp_stock_number)):
		stock_number[i] = float(temp_stock_number[i])
	length_of_stock_number=len(stock_number)

	for i in range(len(stock_pairs_binggo1to9)):
		for j in range(len(stock_pairs_plus10)/length_of_stock_number):
			if (stock_pairs_plus10[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][1]+0.000001)/(stock_pairs_plus10[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][0]+0.000001)>=g1:
				stock_pairs_binggo1to9[i][2]+=1
				stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]=(stock_pairs_plus10[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][4]+0.000001)/(stock_pairs_plus10[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][3]+0.000001)
				if stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]>=g2:
					stock_pairs_binggo1to9[i][3]+=1

	for i in range(len(stock_pairs_binggo1to9)):
		for j in range(len(stock_pairs_plus11)/length_of_stock_number):
			if (stock_pairs_plus11[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][1]+0.000001)/(stock_pairs_plus11[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][0]+0.000001)>=g1:
				stock_pairs_binggo1to9[i][2]+=1
				stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]=(stock_pairs_plus11[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][4]+0.000001)/(stock_pairs_plus11[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][3]+0.000001)
				if stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]>=g2:
					stock_pairs_binggo1to9[i][3]+=1

	for i in range(len(stock_pairs_binggo1to9)):
		for j in range(len(stock_pairs_plus12)/length_of_stock_number):
			if (stock_pairs_plus12[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][1]+0.000001)/(stock_pairs_plus12[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][0]+0.000001)>=g1:
				stock_pairs_binggo1to9[i][2]+=1
				stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]=(stock_pairs_plus12[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][4]+0.000001)/(stock_pairs_plus12[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][3]+0.000001)
				if stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]>=g2:
					stock_pairs_binggo1to9[i][3]+=1

	for i in range(len(stock_pairs_binggo1to9)):
		for j in range(len(stock_pairs_plus1501)/length_of_stock_number):
			if (stock_pairs_plus1501[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][1]+0.000001)/(stock_pairs_plus1501[stock_pairs_binggo1to9[i][0]+j*length_of_stock_number][0]+0.000001)>=g1:
				stock_pairs_binggo1to9[i][2]+=1
				stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]=(stock_pairs_plus1501[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][4]+0.000001)/(stock_pairs_plus1501[stock_pairs_binggo1to9[i][1]+j*length_of_stock_number][3]+0.000001)
				if stock_pairs_binggo1to9[i][3+stock_pairs_binggo1to9[i][2]]>=g2:
					stock_pairs_binggo1to9[i][3]+=1


	for i in range(len(stock_pairs_binggo1to9)):
		stock_pairs_binggo1to9[i][-1]=1


	for i in range(len(stock_pairs_binggo1to9)):
		for j in range(stock_pairs_binggo1to9[i][2]):

			stock_pairs_binggo1to9 [i][-1] *=((stock_pairs_binggo1to9[i][j+4]-trade_fee)*(stock_pairs_binggo1to9[i][j+4]>=0.98)+0.98*(stock_pairs_binggo1to9[i][j+4]<0.98))

	with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_results1yo9.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(stock_pairs_binggo1to9)):
			writer.writerow(stock_pairs_binggo1to9[i][:])

	print  'OK'


def back_test_result():
	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_results1yo9.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:40])
	binggo_results_1to9= np.zeros((len(temp_stock_pairs_plus), 5), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		binggo_results_1to9[i][0] =float( temp_stock_pairs_plus[i][-1])
	del temp_stock_pairs_plus
	gc.collect()

	binggo_resluts=np.zeros_like(binggo_results_1to9)
	j=0
	for i in range(len(binggo_results_1to9)):
		if binggo_results_1to9[i][0]>0.1 and binggo_results_1to9[i][0]<1.5 :
			binggo_resluts[j][0]=binggo_results_1to9[i][0]
			j=j+1

	plt.plot(binggo_resluts[:,0])
	plt.show()

	plt.hist(binggo_resluts[77000:78734,0],500)
	plt.show()

	g=np.average(binggo_resluts[77000:78734,0])





	print 'OK'





print 'ok'

#back_test_gg(1.05,1.01,0.003)
back_test_result()


print 'OK'
