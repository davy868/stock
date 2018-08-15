#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
#from pylab import imshow, show
from timeit import default_timer as timer
# from numba import cuda
# from numba import *

from numpy import array, zeros, argmin, inf, equal, ndim
from find_pairs import *
from testdtw0223 import *
import operator
import csv
import math
from matplotlib import pyplot
import gc
from sqlalchemy import create_engine
import tushare as ts
import pymysql
import time
import pandas as pd
import re

def get_sql_data(g1plus,g1mins,g2):
	stock_number = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_number.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			stock_number = item

	conn=pymysql.connect(host='127.0.0.1',port=3306,user='root',passwd='Luo0046sq',db='mysql',charset='utf8',cursorclass=pymysql.cursors.DictCursor)
	cur=conn.cursor()
	cur.execute("USE stock_5min_from20151201")
	cur.execute("select date from stock_5min_from20151201.601398 WHERE  date  REGEXP '09:35$';")
	ret= cur.fetchall()

	date_from_sql=[]

	for i in range(267,610):
		date_from_sql.append(str(ret[i].values()[0][0:10]))

	stock_data_from_sql=np.zeros((len(stock_number)*343,20),dtype=np.float32)

	print time.time()
	for i in range(len(stock_number)):
		print i
		if i !=688 and i!=830 and i!=832 and i!=597 :
			one_stock_from_2015=[]
			command="select date,open,close,high,low from stock_5min_from20151201."+stock_number[i]+";"
			cur.execute(command)
			ret=cur.fetchall()
			for t in range(len(ret)):
				one_stock_from_2015.append(ret[t].values())

			m=0
			n=0
			d=0
			for d in range(len(date_from_sql)):
				while one_stock_from_2015[m][0][0:10]!=date_from_sql[d]:
					m +=1
					if m==len(one_stock_from_2015)-1:
						break
				if one_stock_from_2015[m][0][0:10]==date_from_sql[d]:
					break
				m=0
			n=m
			while one_stock_from_2015[m][0][0:10] == date_from_sql[d]:
				m += 1

			# print stock_number[i]," ",n," ",m," ",d

			for j in range(d+1,len(date_from_sql)-1):
				n=m
				if m== len(one_stock_from_2015):
					break
				while one_stock_from_2015[m][0][0:10]==date_from_sql[j]:
					if m-n>48:
						m=n
						break
					else:
						m+=1
					if m==len(one_stock_from_2015):
						break

				if m-n>9 and m-n <= 48:
					one_day_data=np.zeros((m-n,4),dtype=np.float32)
					for t in range(m-n):
						one_day_data[t]=one_stock_from_2015[n+t][1:5]
					stock_data_from_sql[i + j * len(stock_number)][0] = one_stock_from_2015[n-1][1]  # 前一天收盘价
					stock_data_from_sql[i + j * len(stock_number)][1] = one_day_data[0][2]  # 上午开盘价
					stock_data_from_sql[i + j * len(stock_number)][2] = max(one_day_data[0:8,1]) #40分钟内最大值
					stock_data_from_sql[i + j * len(stock_number)][3] = min(one_day_data[0:8,3]) #40分钟内最小值
					stock_data_from_sql[i + j * len(stock_number)][4] = one_day_data[8:10].mean()#买入价
					if 1.005*one_day_data[0].mean()<one_day_data[1].mean():
						stock_data_from_sql[i + j * len(stock_number)][5] = one_day_data[-1].mean()  #卖出价
					else:
						stock_data_from_sql[i + j * len(stock_number)][5] = (one_day_data[2].mean()+one_day_data[3].mean())*0.5 #卖出价
					stock_data_from_sql[i + j * len(stock_number)][6] = (stock_data_from_sql[i + j * len(stock_number)][2] + 0.00001) / (stock_data_from_sql[i + j * len(stock_number)][0] + 0.00001)  # g1+
					stock_data_from_sql[i + j * len(stock_number)][7] = (stock_data_from_sql[i + j * len(stock_number)][3] + 0.00001) / (stock_data_from_sql[i + j * len(stock_number)][0] + 0.00001)  # g1-
					stock_data_from_sql[i + j * len(stock_number)][8] = (stock_data_from_sql[i + j * len(stock_number)][5] + 0.00001) / (stock_data_from_sql[i + j * len(stock_number)][4] + 0.00001)  # g2
					if stock_data_from_sql[i + j * len(stock_number)][6] > g1plus:
						stock_data_from_sql[i + j * len(stock_number)][10] = 1
					if stock_data_from_sql[i + j * len(stock_number)][7] < g1mins:
						stock_data_from_sql[i + j * len(stock_number)][11] = 1
					if stock_data_from_sql[i + j * len(stock_number)][8] > g2:
						stock_data_from_sql[i + j * len(stock_number)][12] = 1

	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_07192000.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for t in range(len(stock_data_from_sql)):
			writer.writerow(stock_data_from_sql[t][:])


	# 	for j in range(1,len(date_from_sql)):
	# 		command="select close from stock_5min_from20151201."+stock_number[i]+" where date REGEXP "+"'"+date_from_sql[j-1]+"';"
	# 		cur.execute(command)
	# 		ret=cur.fetchall()
	# 		if len(ret) !=0:
	# 			stock_data_from_sql[i+j*len(stock_number)][0]=ret[-1].values()[0]  #前天收盘价
	# 		command="select open,close,high,low from stock_5min_from20151201."+stock_number[i]+" where date REGEXP "+"'"+date_from_sql[j]+"';"
	# 		cur.execute(command)
	# 		ret=cur.fetchall()
	# 		if len(ret) > 10 :
	# 			one_day_data=np.zeros((len(ret),4),dtype=np.float32)
	# 			for m in range(len(ret)):
	# 				one_day_data[m]=ret[m].values()
	# 			stock_data_from_sql[i + j * len(stock_number)][1] = one_day_data[0][2] #上午开盘价
	# 			stock_data_from_sql[i + j * len(stock_number)][2] = max(one_day_data[0:8,0]) #40分钟内最大值
	# 			stock_data_from_sql[i + j * len(stock_number)][3] = min(one_day_data[0:8,3]) #40分钟内最小值
	# 			stock_data_from_sql[i + j * len(stock_number)][4] = 0.5*np.average(one_day_data[8:10,0])+0.5*np.average(one_day_data[8:10,3])#买入价
	# 			if 1.005*np.average(ret[0].values())<np.average(ret[1].values()):
	# 				stock_data_from_sql[i + j * len(stock_number)][5] = ret[-1].values()[0]  #卖出价
	# 			else:
	# 				stock_data_from_sql[i + j * len(stock_number)][5] = (ret[2].values()[0]+ret[3].values()[0])*0.5   #卖出价
	# 	if i==3:
	# 		with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_1.csv", "w") as csvfile:
	# 			writer = csv.writer(csvfile)
	# 			for t in range(len(stock_data_from_sql)):
	# 				writer.writerow(stock_data_from_sql[t][:])
	# 	if i==500:
	# 		with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_2.csv", "w") as csvfile:
	# 			writer = csv.writer(csvfile)
	# 			for t in range(len(stock_data_from_sql)):
	# 				writer.writerow(stock_data_from_sql[t][:])
	# 	if i==750:
	# 		with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_3.csv", "w") as csvfile:
	# 			writer = csv.writer(csvfile)
	# 			for t in range(len(stock_data_from_sql)):
	# 				writer.writerow(stock_data_from_sql[t][:])
	# 	if i==927:
	# 		with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_4.csv", "w") as csvfile:
	# 			writer = csv.writer(csvfile)
	# 			for t in range(len(stock_data_from_sql)):
	# 				writer.writerow(stock_data_from_sql[t][:])
	print time.time()
	conn.close()
	cur.close()

def get_base_G_from_sql(g1plus,g1mins,g2):
	temp_eigenValue_G = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_all.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_eigenValue_G.append(item[0:20])
	eigenValue_G = np.zeros((len(temp_eigenValue_G), 20), np.float32)
	for i in range(len(temp_eigenValue_G)):
		eigenValue_G[i][0:20] = map(np.float32, temp_eigenValue_G[i][0:20])  ##从csv文件中取数
	for i in range(len(eigenValue_G)):
		eigenValue_G[i][6] = (eigenValue_G[i][2] + 0.00001) / (eigenValue_G[i][0] + 0.00001)  # g1+
		eigenValue_G[i][7] = (eigenValue_G[i][3] + 0.00001) / (eigenValue_G[i][0] + 0.00001)  # g1-
		eigenValue_G[i][8] = (eigenValue_G[i][5] + 0.00001) / (eigenValue_G[i][4] + 0.00001)  # g2

		if eigenValue_G[i][6] > g1plus:
			eigenValue_G[i][10] = 1
		if eigenValue_G[i][7] < g1mins:
			eigenValue_G[i][11] = 1
		if eigenValue_G[i][8] > g2:
			eigenValue_G[i][12] = 1

	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_real.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(eigenValue_G)):
			writer.writerow(eigenValue_G[i][:])

##get_base_G_from_sql(1.06,0.85,1.005)
#get_sql_data(1.06,0.85,1.005)

def back_test_for_test(g1plus,g2,trade_fee):
	temp_test_base_G = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_data_from_sql_base_G_07192000.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_test_base_G.append(item[0:20])
	back_test_G = np.zeros((len(temp_test_base_G), 20), np.float32)
	for i in range(len(temp_test_base_G)):
		back_test_G[i][0:20] = map(np.float32, temp_test_base_G[i][0:20])  ##从csv文件中取数
	del temp_test_base_G
	gc.collect()


	temp_plus_pairs = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_from_sql_2311.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_plus_pairs.append(item[0:60])
	test_plus_pairs = np.zeros((len(temp_plus_pairs), 10), np.float32)
	for i in range(len(temp_plus_pairs)):
		test_plus_pairs[i][0:4] = map(np.float32, temp_plus_pairs[i][0:4])  ##stockA stockB g1计数 g2计数
		test_plus_pairs[i][4] = float(temp_plus_pairs[i][55]) #联乘积
		test_plus_pairs[i][5]=test_plus_pairs[i][3]/test_plus_pairs[i][2]  #g2比率
		test_plus_pairs[i][6]=pow(test_plus_pairs[i][4],1/(test_plus_pairs[i][2]))#几何收益均值
		test_plus_pairs[i][7]=test_plus_pairs[i][6]*(test_plus_pairs[i][5]>0.75)#g2>0.75且几何平均值
	del temp_plus_pairs
	gc.collect()
	temp_bingo_index = np.argsort(test_plus_pairs[:,7])
	temp_bingo_value = np.sort(test_plus_pairs[:,7])


	sorted_valued_pairs=test_plus_pairs[temp_bingo_index[178990:]]

	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_hello_world.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(sorted_valued_pairs)):
			writer.writerow(sorted_valued_pairs[i][:])


	back_test_results=np.zeros((len(sorted_valued_pairs)*2,60), np.float32)
	for i in range(len(sorted_valued_pairs)):
		m=i
		back_test_results[2*m][0]=int(sorted_valued_pairs[i][0])
		back_test_results[2*m+1][0] = int(sorted_valued_pairs[i][1])
		back_test_results[2*m+1][59]=1
		for j in range(343):
			if back_test_G[int(back_test_results[2*m][0]+j*928)][6]>g1plus:
				back_test_results[2*m][1]+=1  #g1plus 计数
				back_test_results[2*m][int(back_test_results[2*m][1]+1)]=back_test_G[int(back_test_results[2*m][0]+j*928)][6]  #g1plus
				temp_value=back_test_G[int(back_test_results[2*m+1][0] + j * 928)][6]  #g2
				back_test_results[2*m+1][int(back_test_results[2*m][1]+1)]=temp_value*(temp_value>0)+(temp_value==0)  #g2,为0时==1
				back_test_results[2*m+1][1]+=(back_test_G[int(back_test_results[2*m+1][0] + j * 928)][6]>g2)  #g2计数

	for i in range(len(back_test_results)/2):
		for t in range(int(back_test_results[2*i][1])):
			back_test_results[2*i+1][59]*=(back_test_results[2*i+1][t+2]-trade_fee)
		back_test_results[2*i+1][58]=pow(back_test_results[2*i+1][59],1/back_test_results[2*i][1])



	# plt.plot(back_test_results[1:-1:2,58])
	# plt.show()
	# mean=[]
	# for i in range(101):
	# 	mean.append(np.mean(back_test_results[(10*(i-1)+1):(10*i+1):2,58]))
	# plt.figure(2)
	# plt.plot(mean[:])
	# plt.show()
	#
	# print np.average(mean[1:])
	# print np.sum(back_test_results[0:-1:2,1])

	print "OK"



back_test_for_test(1.05,1.005,0.003)


print  'OK'


