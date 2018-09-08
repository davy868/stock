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
import time




def get_file(inputfile):
###################################################
#  输入目录及子目录下所有的股票分钟数据转化为
#  日期数 g1 g2
#  使用pool = Pool(processes=4)，pool.map(temp_pool,tasks)
#   执行多核运算，最终分别保存到四个文件中
###################################################

	temp_stock_number = []
	with open("/root/PycharmProjects/untitled/stock_number.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_number = item
	stock_number = np.zeros(len(temp_stock_number), float)
	for i in range(len(temp_stock_number)):
		stock_number[i] = float(temp_stock_number[i])


	all_stock_data1=np.zeros((len(stock_number)*125,15), dtype=np.float32)
	all_stock_data2 = np.zeros((len(stock_number) * 125, 15), dtype=np.float32)
	all_stock_data3 = np.zeros((len(stock_number) * 125, 15), dtype=np.float32)
	all_stock_data4 = np.zeros((len(stock_number) * 125, 15), dtype=np.float32)


	filedirs = os.listdir(inputfile)
	temp_filedir_order = np.zeros(len(filedirs), dtype=np.int)
	filedirs_sorted = list()


##########################
#将输入目录下的交易日数据按照时间的先后排序，输出filedirs_sorted
	for i in range(len(filedirs)):
		temp_filedir_order[i] = int(filedirs[i][0:6])
	temp_file_order_sorted = np.argsort(temp_filedir_order)
	for i in range(len(filedirs)):
		filedirs_sorted.append(inputfile+'/'+filedirs[temp_file_order_sorted[i]])
#########################
	filedirs_sorted_day = list()
	for i in range(len(filedirs_sorted)):
		filedirs = os.listdir(filedirs_sorted[i])
		temp_filedir_order = np.zeros(len(filedirs), dtype=np.int)


		##########################
		# 将输入目录下的交易日数据按照时间的先后排序，输出filedirs_sorted
		for j in range(len(filedirs)):
			temp_filedir_order[j] = int(filedirs[j][5:11])
		temp_file_order_sorted = np.argsort(temp_filedir_order)
		for j in range(len(filedirs)):
			filedirs_sorted_day.append(filedirs_sorted[i] + '/' + filedirs[temp_file_order_sorted[j]])

	length_of_file=len(filedirs_sorted_day)
	tasks=[(filedirs_sorted_day[0:int(length_of_file*0.25)],stock_number,all_stock_data1),(filedirs_sorted_day[int(length_of_file*0.25):int(length_of_file*0.25)*2],stock_number,all_stock_data2),(filedirs_sorted_day[int(length_of_file*0.25)*2:int(length_of_file*0.25)*3],stock_number,all_stock_data3),(filedirs_sorted_day[int(length_of_file*0.25)*3:length_of_file],stock_number,all_stock_data4)]
	pool = Pool(processes=4)
	pool.map(temp_pool,tasks)
	# get_all_stock_data(tasks[0][0],tasks[0][1],tasks[0][2])

	return filedirs_sorted_day

def get_all_stock_data(filedirs_sorted,stock_number,all_stock_data):
	length_of_stock_number=len(stock_number)
	base_info_stock=np.zeros((length_of_stock_number,len(filedirs_sorted)*3),dtype=np.float32)


	for i in range(0, len(filedirs_sorted)):

		day_data_in_one_array = get_day_data(filedirs_sorted[i], stock_number)#读取某一天的数据，股票代码为矩阵行号
		all_stock_data[i * length_of_stock_number:(i + 1) * length_of_stock_number,0:11] = day_data_in_one_array[:,0:11]#将某天的数据放入大矩阵all_stock_data中
		all_stock_data[i * length_of_stock_number:(i + 1) * length_of_stock_number, 11:15] = day_data_in_one_array[:,-4:]  # 将某天的收盘价放入all_stock_data中
		if i>1 and i<len(filedirs_sorted)-1 :
			for j in range(length_of_stock_number):
				base_info_stock[j][(i-1)*3]=(time.mktime(time.strptime(filedirs_sorted[i-1][-24:-16],'%Y%m%d'))-time.mktime(time.strptime('2014-01-01','%Y-%m-%d')))/(24*3600)
				base_info_stock[j][(i-1)*3+1]=(all_stock_data[(i-1)*length_of_stock_number+j][0]+0.00001)/(all_stock_data[(i-2)*length_of_stock_number+j][-1]+0.00001)#g1:今天开盘价/昨天收盘价
				base_info_stock[j][(i-1)*3+2]=(np.average(all_stock_data[(i)*length_of_stock_number+j][1:6])+0.00001)/(np.average(all_stock_data[(i-1)*length_of_stock_number+j][6:11])+0.00001)#g2:明天卖出价/今天买入价
	with open("/root/PycharmProjects/untitled/base_info_"+filedirs_sorted[0][-24:-4]+".csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(base_info_stock)):
			writer.writerow(base_info_stock[i][:])
def temp_pool(args):
	get_all_stock_data(args[0],args[1],args[2])


def get_day_data(filedir,stock_number):
	day_prices_in_one_array = get_day_prices_in_one_array(filedir)
	day_prices_in_one_array.astype(np.float32)
	for i in range(0,len(day_prices_in_one_array)):
		for j in range(len(stock_number)):
			if stock_number[j] == day_prices_in_one_array[i][243] :
				day_prices_in_one_array[i][243]=j+1  #避开0
				break

	day_data_in_order=np.zeros((len(stock_number),243),dtype=np.float32)
	for i in range(len(stock_number)):
		if day_prices_in_one_array[i,243]>0 and day_prices_in_one_array[i,243]<(len(stock_number)+1):
			day_data_in_order[int(day_prices_in_one_array[i,243])-1]=day_prices_in_one_array[i][0:243] #-1与上文j+1对应

	return   day_data_in_order


#get_file('/root/PycharmProjects/untitled/minute_data')

def combine_all_base_info():
	file1 = "/root/PycharmProjects/untitled/base_info_20140102_minute_data.csv"
	file2 = "/root/PycharmProjects/untitled/base_info_20140616_minute_data.csv"
	file3 = "/root/PycharmProjects/untitled/base_info_20141121_minute_data.csv"
	file4 = "/root/PycharmProjects/untitled/base_info_20150519_minute_data.csv"
	file5 = "/root/PycharmProjects/untitled/base_info_stock_gg_1808282148.csv"

	base_info_all_stock=np.zeros((928,1100*3),dtype=np.float32)


	temp_all_stock_number = []
	with open(file1, "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		base_info_all_stock[i,0:324] = map(np.float32,temp_all_stock_number[i])
	print "file1 ok"

	temp_all_stock_number = []
	with open(file2, "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		base_info_all_stock[i, 324:324*2] = map(np.float32, temp_all_stock_number[i])
	print "file2 ok"

	temp_all_stock_number = []
	with open(file3, "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		base_info_all_stock[i, 324*2:324*3] = map(np.float32, temp_all_stock_number[i])
	print "file3 ok"


	temp_all_stock_number = []
	with open(file4, "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		base_info_all_stock[i, 324*3:(324*3+330)] = map(np.float32, temp_all_stock_number[i])
	print "file4 ok"


	temp_all_stock_number = []
	with open(file5, "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		base_info_all_stock[i, (324*3+330):(324*3+330+1830)] = map(np.float32, temp_all_stock_number[i])
	print "file5 ok"

	with open("/root/PycharmProjects/untitled/base_info_all_stock.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(base_info_all_stock)):
			writer.writerow(base_info_all_stock[i][:])

	print "OK"

# combine_all_base_info()

def correct_base_info():
	temp_all_stock_number = []
	with open('/root/PycharmProjects/untitled/base_info_all_stock.csv', "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	base_all=np.zeros((928,len(temp_all_stock_number[0])),dtype=np.float32)
	for i in range(len(temp_all_stock_number)):
		base_all[i,:] = map(np.float32, temp_all_stock_number[i])
		for j in range(len(base_all[i])):
			if j%3==1:
				if base_all[i][j]>0.89 and base_all[i][j]<1.11 :
					temp=i
				else:
					base_all[i][j]=1
			if j%3==2:
				if base_all[i][j]>0.79 and base_all[i][j]<1.3 :
					temp=i
				else:
					base_all[i][j]=1


	with open("/root/PycharmProjects/untitled/base_info_all_stock_correction.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(base_all)):
			writer.writerow(base_all[i][:])

def produce_stock_pairs(g1set,g2set):
	temp_all_stock_number = []
	with open('/root/PycharmProjects/untitled/base_info_all_stock_correction.csv', "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	base_all = np.zeros((928, len(temp_all_stock_number[0])), dtype=np.float32)
	for i in range(len(temp_all_stock_number)):
		base_all[i, :] = map(np.float32, temp_all_stock_number[i])

	base_stock_pairs=np.zeros((928*928,420),dtype=np.float32)
	for i in range(928*928):

		base_stock_pairs[i][0]=i/928
		base_stock_pairs[i][1]=i%928

	for i in range(928):
		for j in range(1100):
			if base_all[i][j*3] != 0 :
				if base_all[i][j*3+1]>= g1set:
					for t in range(928):
						base_stock_pairs[928*i+t][3*int(base_stock_pairs[928*i+t][2])+4]=base_all[i][j*3]
						base_stock_pairs[928 * i + t][3*int(base_stock_pairs[928 * i + t][2]) + 5] = base_all[i][j * 3+1]
						base_stock_pairs[928 * i + t][3*int(base_stock_pairs[928 * i + t][2]) + 6] = base_all[t][j * 3 + 2]
						base_stock_pairs[928 * i + t][2]+=1
						if base_all[t][j * 3 + 2]>=g2set:
							base_stock_pairs[928 * i + t][3] += 1
	with open("/root/PycharmProjects/untitled/base_stock_pairs.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(base_stock_pairs)):
			writer.writerow(base_stock_pairs[i][:])
# correct_base_info()

# produce_stock_pairs(1.01,1.02)


# temp_stock_number = []
# with open("/root/PycharmProjects/untitled/stock_number.csv", "r") as csvfile:
# 	reader = csv.reader(csvfile)
# 	for item in reader:
# 		temp_stock_number = item
# stock_number = np.zeros(len(temp_stock_number), float)
# for i in range(len(temp_stock_number)):
# 	stock_number[i] = float(temp_stock_number[i])

def compute_geomean(ratio_n2_n1,trade_fee):
	temp_all_stock_number = []
	with open('/root/PycharmProjects/untitled/base_stock_pairs.csv', "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			if float(item[3])/float(item[2]) >= ratio_n2_n1:
				temp_all_stock_number.append(item)
	base_all = np.zeros((len(temp_all_stock_number), len(temp_all_stock_number[0])), dtype=np.float32)
	for i in range(len(temp_all_stock_number)):
		base_all[i, :] = map(np.float32, temp_all_stock_number[i])
		base_all[i,-2]=1
		for j in range(2,int(len(base_all[i,:])/3)):
			if base_all[i,3*j] !=0 :
				base_all[i,-2] *= (base_all[i,3*j]-trade_fee)
		base_all[i,-1]=pow(base_all[i,-2],1/base_all[i,2])

	print  "OK"




compute_geomean(0.5,0.002)
print "OK"