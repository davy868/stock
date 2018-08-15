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

def get_sorted_axle(inputfile):
	#####################################
	# 获取输入目录下所有股票的代码，保存在stock_number中和stock_number.csv文件中
	# 按照升序排列，即600000,600001，...
	# 假定股票总数不超过1000，输入目录下最多有25个交易日，每只交易数据长度为244，最后一个为股票代码
	#####################################
	temp_stock_number = np.zeros((1000*25), dtype=np.float32)
	filedirs = os.listdir(inputfile)
	for i in range(0, len(filedirs)):
		filedirs[i] = inputfile + filedirs[i]
		day_prices_in_one_array = get_day_prices_in_one_array(filedirs[i])
		day_prices_in_one_array.astype(np.float32)
		temp_stock_number[i*1000:(i+1)*1000]=day_prices_in_one_array[:,243]
		# np.append(temp_stock_number,day_prices_in_one_array[:,243])
	stock_number=np.unique(temp_stock_number)
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_number.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(stock_number[1:])
	return stock_number[1:]


def get_base_G(inputfile,g1plus,g1mins,g2):
##############################################
#获取输入目录下一个月内每一只股票的上午开盘价、上午最高价、上午最低价、下午开盘价、第二天收盘价
#结果为矩阵base_G(所有股票数*当月交易日数×20)，并保存为base_G.csv
##############################################
	#stock_number = get_sorted_axle(inputfile)


	temp_stock_number = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_number.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_number=item
	stock_number = np.zeros(len(temp_stock_number), float)
	for i in range(len(temp_stock_number)):
		stock_number[i] = float(temp_stock_number[i])



	filedirs = os.listdir(inputfile)
	temp_filedir_order = np.zeros(len(filedirs), dtype=np.int)
	length_of_stock_number = len(stock_number)
	all_stock_data = np.zeros((length_of_stock_number * len(filedirs), 243), dtype=np.float32)
	all_stock_data1 = np.zeros((89088, 243), dtype=np.float32)
	all_stock_data2 = np.zeros((89088, 243), dtype=np.float32)
	all_stock_data3 = np.zeros((89088, 243), dtype=np.float32)
	all_stock_data4 = np.zeros((90944, 243), dtype=np.float32)
	base_G = np.zeros((length_of_stock_number * (len(filedirs)-2), 20), dtype=np.float32)#一个月中最后一天的数据不全用
	filedirs_sorted = list()


##########################
#将输入目录下的交易日数据按照时间的先后排序，输出filedirs_sorted
	for i in range(len(filedirs)):
		temp_filedir_order[i] = int(filedirs[i][5:11])
	temp_file_order_sorted = np.argsort(temp_filedir_order)
	for i in range(len(filedirs)):
		filedirs_sorted.append(filedirs[temp_file_order_sorted[i]])
#########################

	tasks=[(inputfile,filedirs_sorted[0:int(len(filedirs)*0.25)],stock_number,all_stock_data1),(inputfile,filedirs_sorted[int(len(filedirs)*0.25):int(len(filedirs)*0.25)*2],stock_number,all_stock_data2),(inputfile,filedirs_sorted[int(len(filedirs)*0.25)*2:int(len(filedirs)*0.25)*3],stock_number,all_stock_data3),(inputfile,filedirs_sorted[int(len(filedirs)*0.25)*3:len(filedirs_sorted)],stock_number,all_stock_data4)]
	# pool = Pool(processes=4)
	# pool.map(temp_pool,tasks)

	# for i in range(0, len(filedirs_sorted)):
	# 	filedirs_sorted[i] = inputfile + filedirs_sorted[i]
	# 	day_data_in_one_array = get_day_data(filedirs_sorted[i], stock_number)#读取某一天的数据，股票代码为矩阵行号
	# 	all_stock_data[i * length_of_stock_number:(i + 1) * length_of_stock_number][:] = day_data_in_one_array#将某天的数据放入大矩阵all_stock_data中

	# 以下为测试代码，测试all_stock_data的正确性
	# day_prices_in_one_array = get_day_prices_in_one_array(inputfile+'sh_20140114_minute_data.csv')
	# plt.figure(1)
	# plt.plot(day_prices_in_one_array[9][0:243])
	# plt.figure(2)
	# plt.plot(all_stock_data[8*length_of_stock_number+54][:])
	# plt.show()
	temp_all_stock_number=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/all_stock_0102_minute_data.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number[:])):
		all_stock_data[i] = map(np.float32,temp_all_stock_number[i])


	temp_all_stock_number = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/all_stock_0313_minute_data.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		all_stock_data[i+len(temp_all_stock_number)] = map(np.float32,temp_all_stock_number[i])


	temp_all_stock_number = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/all_stock_0529_minute_data.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		all_stock_data[i+len(temp_all_stock_number)*2] = map(np.float32,temp_all_stock_number[i])

	temp_all_stock_number=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/all_stock_1021_minute_data.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_all_stock_number.append(item)
	for i in range(len(temp_all_stock_number)):
		all_stock_data[i+len(temp_all_stock_number)*3] = map(np.float32,temp_all_stock_number[i])

	del temp_all_stock_number
	gc.collect()
	#
	#
	# flucation_stock_data=[]
	# for i in range(len(all_stock_data)):
	# 	max=np.max(all_stock_data[i,:])
	# 	min=np.min(all_stock_data[i,:])
	# 	avgr=np.average(all_stock_data[i,:])
	# 	if (max-min+0.0000001)/(avgr+0.001) >0.01 :
	# 		flucation_stock_data.append(all_stock_data[i])
	#
	# real_flucation_stock_data=np.zeros_like(flucation_stock_data)
	# for i in range(len(flucation_stock_data)):
	# 	real_flucation_stock_data[i]=map(np.float32,flucation_stock_data[i])
	#
	# index_all_stock_data=np.zeros_like(real_flucation_stock_data)
	# for i in range(len(real_flucation_stock_data)):
	# 	index_all_stock_data[i]=np.argsort(real_flucation_stock_data[i])
	# plt.hist(index_all_stock_data[:,-1],100,normed=True)
	# plt.show()
	#
	# plt.figure(2)
	# plt.hist(index_all_stock_data[:,0],100,normed=True)
	# plt.show()
	#
	#
	# plt.figure(3)
	# plt.hist(index_all_stock_data[:,122],100,normed=True)
	# plt.show()




	for i in range(length_of_stock_number,len(base_G)):
		base_G[i][0] = all_stock_data[i-length_of_stock_number][-1] #昨天收盘价
		base_G[i][1] = all_stock_data[i][0] #上午开盘价
		base_G[i][2] = np.max(all_stock_data[i][0:40])  # 上午最高价
		base_G[i][3] = np.min(all_stock_data[i][0:40])  # 上午最低价
		base_G[i][4] = np.average(all_stock_data[i][40:50])  # 下单价

		if 1.005*(np.average(all_stock_data[i + length_of_stock_number][0:5])) < np.average(all_stock_data[i + length_of_stock_number][5:10]) :
			base_G[i][5]= np.average(all_stock_data[i + length_of_stock_number][239:])
		else:
			base_G[i][5]=np.average(all_stock_data[i + length_of_stock_number][10:20])

		base_G[i][6]=(base_G[i][2]+0.00001)/(base_G[i][0]+0.00001)  #g1+
		base_G[i][7]=(base_G[i][3]+0.00001)/(base_G[i][0]+0.00001) #g1-
		base_G[i][8]=(base_G[i][5]+0.00001)/(base_G[i][4]+0.00001) #g2

		if base_G[i][6]>g1plus:
			base_G[i][10]=1
		if base_G[i][7]<g1mins:
			base_G[i][11]=1
		if base_G[i][8]>g2:
			base_G[i][12]=1


	with open("/root/neural-networks-and-deep-learning/csv5minute/base_G.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(base_G)):
			writer.writerow(base_G[i][:])
	return base_G

def get_all_stock_data(inputfile,filedirs_sorted,stock_number,all_stock_data):
	length_of_stock_number=len(stock_number)
	for i in range(0, len(filedirs_sorted)):
		filedirs_sorted[i] = inputfile + filedirs_sorted[i]
		day_data_in_one_array = get_day_data(filedirs_sorted[i], stock_number)#读取某一天的数据，股票代码为矩阵行号
		all_stock_data[i * length_of_stock_number:(i + 1) * length_of_stock_number][:] = day_data_in_one_array#将某天的数据放入大矩阵all_stock_data中
	with open("/root/neural-networks-and-deep-learning/csv5minute/all_stock_"+filedirs_sorted[0][-20:-4]+".csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(all_stock_data)):
			writer.writerow(all_stock_data[i][:])
def temp_pool(args):
	get_all_stock_data(args[0],args[1],args[2],args[3])

# #get_base_G(inputfile)
# def get_eigenValue_G(g1plus,g1minus,g2):
# ################################################
# #首先计算每个交易日，每支股票的g1+,g1-和g2，然后根据输入的g1+,g1-和g2的阈值，将相应的标志位置1
# ################################################
# 	temp_eigenValue_G=[]
# 	with open("/root/neural-networks-and-deep-learning/csv5minute/base_G.csv", "r") as csvfile:
# 		reader = csv.reader(csvfile)
# 		for item in reader:
# 			temp_eigenValue_G.append(item[0:20])
# 	eigenValue_G=np.zeros((len(temp_eigenValue_G),20),np.float32)
# 	for i in range(len(temp_eigenValue_G)):
# 		eigenValue_G[i][0:20]=map(float32,temp_eigenValue_G[i][0:20])
#
#
# 	for i in range(len(eigenValue_G)):
# 		eigenValue_G[i][6]=(eigenValue_G[i][1]+0.000001)/(eigenValue_G[i][0]+0.000001)#g1+,加上极小数是为了规避除0
# 		eigenValue_G[i][7]=(eigenValue_G[i][0]+0.000001)/(eigenValue_G[i][2]+0.000001)#g1-
# 		eigenValue_G[i][8] = (eigenValue_G[i][4] + 0.000001) / (eigenValue_G[i][3] + 0.000001)#g2
#
#
# 		if eigenValue_G[i][6]>=g1plus:
# 			eigenValue_G[i][10]=1    #g1+标志位
# 		if eigenValue_G[i][7]>=g1minus:
# 			eigenValue_G[i][11]=1    #g1-标志位
# 		if eigenValue_G[i][8]>=g2:
# 			eigenValue_G[i][12]=1   #g2标志位
#
# 	#################
#
# 	with open("/root/neural-networks-and-deep-learning/csv5minute/eigenValue_G.csv", "w") as csvfile:
# 		writer = csv.writer(csvfile)
# 		for i in range(len(eigenValue_G)):
# 			writer.writerow(eigenValue_G[i][:])

def get_pairs(length_of_stock_number,inputfile_month,trade_fee,bingo_N):
####################################################
#计算每双配对股票的配对数，保存在stock_pairs_plus和stock_pairs_minus中
#共有length_of_stock_number*length_of_stock_number个对数
#矩阵stock_pairs中每一行代表一个配对，分别是No1(0)，No2(1)，g1+或者g1-(2),g2(3),每个g2的实际数值，共有（g1+或者g1-个）
####################################################
	temp_eigenValue_G=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/base_G_from_csv_1905.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_eigenValue_G.append(item[0:20])
	eigenValue_G=np.zeros((len(temp_eigenValue_G),20),np.float32)
	for i in range(len(temp_eigenValue_G)):
		eigenValue_G[i][0:20]=map(np.float32,temp_eigenValue_G[i][0:20])##从csv文件中取数
	del temp_eigenValue_G
	gc.collect()

	stock_pairs_plus=np.zeros((length_of_stock_number*length_of_stock_number,60),dtype=np.float32)
	stock_pairs_minus = np.zeros((length_of_stock_number * length_of_stock_number, 60), dtype=np.float32)

	for i in range(len(eigenValue_G)):
		if eigenValue_G[i][10]==1:  #g1+
			for t in range((i%length_of_stock_number)*length_of_stock_number,((i%length_of_stock_number)+1)*length_of_stock_number):
				stock_pairs_plus[t][2]+=1#相应的股票位置处加1
			for j in range(int(np.floor(i/length_of_stock_number)*length_of_stock_number),int((np.floor(i/length_of_stock_number)+1)*length_of_stock_number)):
				temp_location=(i%length_of_stock_number)*length_of_stock_number+(j%length_of_stock_number)
				if eigenValue_G[j][8]>1.21 or eigenValue_G[j][8]<0.81 :  #存在股价非市场因素重大变化
					stock_pairs_plus[temp_location][int(stock_pairs_plus[temp_location][2] + 3)]=1
				else:
					stock_pairs_plus[temp_location][int(stock_pairs_plus[temp_location][2]+3)]=eigenValue_G[j][8]
				if eigenValue_G[j][12]==1:  #g2
					stock_pairs_plus[(i%length_of_stock_number)*length_of_stock_number+(j%length_of_stock_number)][3]+=1


		# if eigenValue_G[i][11]==1:#g1-
		# 	for t in range((i%length_of_stock_number)*length_of_stock_number,((i%length_of_stock_number)+1)*length_of_stock_number):
		# 		stock_pairs_minus[t][2]+=1#相应的股票位置处加1
		# 	for j in range(int(np.floor(i/length_of_stock_number)*length_of_stock_number),int((np.floor(i/length_of_stock_number)+1)*length_of_stock_number)):
		# 		temp_location=(i%length_of_stock_number)*length_of_stock_number+(j%length_of_stock_number)
		# 		if eigenValue_G[j][8] > 1.21 or eigenValue_G[j][8] < 0.81:#存在股价非市场因素重大变化
		# 			stock_pairs_minus[temp_location][int(stock_pairs_minus[temp_location][2] + 3)] =1
		# 		else:
		# 			stock_pairs_minus[temp_location][int(stock_pairs_minus[temp_location][2]+3)]=eigenValue_G[j][8]
		# 		if eigenValue_G[j][12]==1:#g2
		# 			stock_pairs_minus[(i%length_of_stock_number)*length_of_stock_number+(j%length_of_stock_number)][3]+=1
		# # if eigenValue_G[i][11]==1:
		# # 	for t in range((i%length_of_stock_number)*length_of_stock_number,((i%length_of_stock_number)+1)*length_of_stock_number):
		# # 		stock_pairs[t][4]+=1
		# # 	for j in range(int(np.floor(i/length_of_stock_number)*length_of_stock_number),int((np.floor(i/length_of_stock_number)+1)*length_of_stock_number)):
		# # 		if eigenValue_G[j][12]==1:
		# # 			stock_pairs[(i%length_of_stock_number)*length_of_stock_number+(j%length_of_stock_number)][5]+=1
	for i in range(len(stock_pairs_plus)):#填充每一行代表的配对股票代码
		stock_pairs_plus[i][0]=np.floor(i/length_of_stock_number)
		stock_pairs_plus[i][1]=i%length_of_stock_number
		stock_pairs_minus[i][0]=np.floor(i/length_of_stock_number)
		stock_pairs_minus[i][1]=i%length_of_stock_number


	for i in range(len(stock_pairs_plus)):
		for t in range(int(stock_pairs_plus[i][2])):
			stock_pairs_plus[i][t + 4] -= trade_fee
			stock_pairs_plus[i][55] = 1
	for i in range(len(stock_pairs_plus)):
		for t in range(int(stock_pairs_plus[i][2])):
			stock_pairs_plus[i][55] *= stock_pairs_plus[i][t + 4]

	temp_plus_pairs_from_sql=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_from_sql_1832.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_plus_pairs_from_sql.append(item[0:60])
		plus_pairs_from_sql=np.zeros((len(temp_plus_pairs_from_sql),60),np.float32)
	for i in range(0,len(temp_plus_pairs_from_sql)):
		plus_pairs_from_sql[i][0:60]=map(np.float32,temp_plus_pairs_from_sql[i][0:60])##从csv文件中取数
	del temp_plus_pairs_from_sql
	gc.collect
	for u in range(len(stock_pairs_plus)):
		stock_pairs_plus[u][2]=stock_pairs_plus[u][2]+plus_pairs_from_sql[u][2]
		stock_pairs_plus[u][3] = stock_pairs_plus[u][3] + plus_pairs_from_sql[u][3]
		stock_pairs_plus[u][55]=stock_pairs_plus[u][55]*plus_pairs_from_sql[u][55]




	#temp_bingo_value = np.sort(stock_pairs_plus[:, 55])
	temp_bingo_index = np.argsort(stock_pairs_plus[:, 55])




	with open(inputfile_month+"stock_pairs_from_sql_2311.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in stock_pairs_plus[temp_bingo_index[int(-1 * bingo_N):], :]:
			writer.writerow(row)
	# with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_from_sql_2159.csv", "w") as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	for t in range(len(stock_pairs_plus))
	# 		writer.writerow(stock_pairs_plus[t][:])

	# writer = csv.writer(csvfile)
		# for i in range(len(stock_pairs_plus)):
		# 	writer.writerow(stock_pairs_plus[i][:])
	# with open(inputfile_month[0:-4]+'realminus.csv', "w") as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	for i in range(len(stock_pairs_minus)):
	# 		writer.writerow(stock_pairs_minus[i][:])

def statistic_pairs_plus(binggo_N,trade_fee):
#################################################
#获取所有配对股票的数据，然后在No1股票满足g1+时，计算所有g2的联乘积
#联乘之前，每个乘数减去交易费用，即trade_fee.
#然后对联乘排序，取出最大的binggo_N个
##################################################
	temp_stock_pairs_plus=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_plus.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:35])
	stock_pairs_plus=np.zeros((len(temp_stock_pairs_plus),35),np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus[i][0:35]=map(np.float32,temp_stock_pairs_plus[i][0:35])##从csv文件中取数
		for t in range(int(stock_pairs_plus[i][2])):
			stock_pairs_plus[i][t+4]-=trade_fee
			stock_pairs_plus[i][25]=1
	for i in range(len(stock_pairs_plus)):
		for t in range(int(stock_pairs_plus[i][2])):
			stock_pairs_plus[i][25] *= stock_pairs_plus[i][t+4]

	temp_bingo_value = np.sort(stock_pairs_plus[:, 25])
	temp_bingo_index = np.argsort(stock_pairs_plus[:, 25])

	with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_plus_stock_pairs.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in stock_pairs_plus[temp_bingo_index[int(-1*binggo_N):], :]:
			writer.writerow(row)


def statistic_pairs_minus(binggo_N,trade_fee):
	#################################################
	# 获取所有配对股票的数据，然后在No1股票满足g1-时，计算所有g2的联乘积
	# 联乘之前，每个乘数减去交易费用，即trade_fee.
	# 然后对联乘排序，取出最大的binggo_N个
	##################################################
	temp_stock_pairs_minus=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_minus.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_minus.append(item[0:35])
	stock_pairs_minus=np.zeros((len(temp_stock_pairs_minus),35),np.float32)
	for i in range(len(temp_stock_pairs_minus)):
		stock_pairs_minus[i][0:35]=map(np.float32,temp_stock_pairs_minus[i][0:35])##从csv文件中取数
		for t in range(int(stock_pairs_minus[i][2])):
			stock_pairs_minus[i][t+4]-=trade_fee
			stock_pairs_minus[i][25]=1
	for i in range(len(stock_pairs_minus)):
		for t in range(int(stock_pairs_minus[i][2])):
			stock_pairs_minus[i][25] *= stock_pairs_minus[i][t+4]

	temp_bingo_value = np.sort(stock_pairs_minus[:, 25])
	temp_bingo_index = np.argsort(stock_pairs_minus[:, 25])

	with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_minus_stock_pairs.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in stock_pairs_minus[temp_bingo_index[int(-1*binggo_N):], :]:
			writer.writerow(row)


def vertification_test_minus(testfile_dirs,length_of_stock_number,g1plus,g1minus):
###########################################
#用输入文件目录下的交易数据验证g1-的排序结果
##########################################
	temp_binggo_stock_pairs_minus=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_minus_stock_pairs.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_binggo_stock_pairs_minus.append(item[0:20])
	binggo_stock_pairs_minus=np.zeros((len(temp_binggo_stock_pairs_minus),20),np.float32)
	for i in range(len(temp_binggo_stock_pairs_minus)):
		binggo_stock_pairs_minus[i][0:20]=map(float32,temp_binggo_stock_pairs_minus[i][0:20])##从csv文件中取数

	vertification_base_G=get_base_G(testfile_dirs)#获取验证数据的基本特征

	length_of_binggo_minus=len(binggo_stock_pairs_minus)

	vertification_minus_result=np.zeros((length_of_binggo_minus,35),dtype=np.float32)#验证集的保留结果
	vertification_minus_result[:,0]=binggo_stock_pairs_minus[:,0]
	vertification_minus_result[:,1]=binggo_stock_pairs_minus[:,1]




	for i in range(length_of_binggo_minus):#计算在满足g1-条件下的各个g2，并保存
		for t in range(int(binggo_stock_pairs_minus[i][0]),len(vertification_base_G),length_of_stock_number):
			if (vertification_base_G[t][0]+0.000001)/(vertification_base_G[t][2]+0.000001)>g1minus:
				vertification_minus_result[i][2]+=1
				vertification_minus_result[i][int(vertification_minus_result[i][2]+3)]=(vertification_base_G[t][4]+0.000001)/(vertification_base_G[t][3]+0.000001)

	for i in range(length_of_binggo_minus):#计算联乘积
		vertification_minus_result[i][25]=1
		for t in range(int(vertification_minus_result[i][2])):
			vertification_minus_result[i][25] *=vertification_minus_result[i][t+4]
	for i in range(length_of_binggo_minus):#如果乘积太小，表明此次配对不可用，乘积结果赋值为1
		if vertification_minus_result[i][25]<0.1:
			vertification_minus_result[i][25]=1

	return vertification_minus_result



def vertification_test_plus(testfile_dirs,length_of_stock_number,g1plus,g2):
	###########################################
	# 用输入文件目录下的交易数据验证g1+的排序结果
	##########################################
	temp_binggo_stock_pairs_plus=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/vertification_result_plus_month9.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_binggo_stock_pairs_plus.append(item[0:20])
	binggo_stock_pairs_plus=np.zeros((len(temp_binggo_stock_pairs_plus),20),np.float32)
	for i in range(len(temp_binggo_stock_pairs_plus)):
		binggo_stock_pairs_plus[i][0:20]=map(float32,temp_binggo_stock_pairs_plus[i][0:20])##从csv文件中取数

	vertification_base_G=get_base_G(testfile_dirs)#获取验证数据的基本特征

	length_of_binggo_plus=len(binggo_stock_pairs_plus)

	vertification_plus_result=np.zeros((length_of_binggo_plus,35),dtype=np.float32)
	vertification_plus_result[:,0]=binggo_stock_pairs_plus[:,0]
	vertification_plus_result[:,1]=binggo_stock_pairs_plus[:,1]


	for i in range(length_of_binggo_plus):#计算在满足g1+条件下的各个g2，并保存
		for t in range(int(binggo_stock_pairs_plus[i][0]),len(vertification_base_G),length_of_stock_number):
			if (vertification_base_G[t][1]+0.000001)/(vertification_base_G[t][0]+0.000001)>g1plus:
				vertification_plus_result[i][2]+=1
				vertification_plus_result[i][int(vertification_plus_result[i][2]+3)]=(vertification_base_G[t][4]+0.000001)/(vertification_base_G[t][3]+0.000001)
				if vertification_plus_result[i][int(vertification_plus_result[i][2]+3)]>g2:
					vertification_plus_result[i][3] += 1

	for i in range(length_of_binggo_plus):#计算联乘积
		vertification_plus_result[i][25]=1
		for t in range(int(vertification_plus_result[i][2])):
			vertification_plus_result[i][25] *=vertification_plus_result[i][t+4]
	for i in range(length_of_binggo_plus):
		if vertification_plus_result[i][25]<0.1:#如果乘积太小，表明此次配对不可用，乘积结果赋值为1
			vertification_plus_result[i][25]=1
	with open("/root/neural-networks-and-deep-learning/csv5minute/vertification_result_plus_month10.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in vertification_plus_result[0:length_of_binggo_plus, :]:
			if row[25]>0.99:
				writer.writerow(row)
	return vertification_plus_result

def together_month1to3_pairs():
#################################################
#获取所有配对股票的数据，然后在No1股票满足g1+时，计算所有g2的联乘积
#联乘之前，每个乘数减去交易费用，即trade_fee.
#然后对联乘排序，取出最大的binggo_N个
##################################################
	temp_stock_pairs_plus=[]
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_plus_month1.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:35])
	stock_pairs_plus_month1=np.zeros((len(temp_stock_pairs_plus),35),np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus_month1[i][0:35]=map(float32,temp_stock_pairs_plus[i][0:35])##从csv文件中取数

	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_plus_month2.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:35])
	stock_pairs_plus_month2 = np.zeros((len(temp_stock_pairs_plus), 35), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus_month2[i][0:35] = map(float32, temp_stock_pairs_plus[i][0:35])  ##从csv文件中取数

	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_plus_month3.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:35])
	stock_pairs_plus_month3 = np.zeros((len(temp_stock_pairs_plus), 35), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus_month3[i][0:35] = map(float32, temp_stock_pairs_plus[i][0:35])  ##从csv文件中取数


	stock_pairs_plus_1to3=np.zeros((len(stock_pairs_plus_month1),55),dtype=np.float32)

	for i in range(len(stock_pairs_plus_1to3)):
		stock_pairs_plus_1to3[i][0:2]=stock_pairs_plus_month1[i][0:2]
		stock_pairs_plus_1to3[i][2]=stock_pairs_plus_month1[i,2]+stock_pairs_plus_month2[i,2]+stock_pairs_plus_month3[i,2]
		stock_pairs_plus_1to3[i][3]=stock_pairs_plus_month1[i,3]+stock_pairs_plus_month2[i,3]+stock_pairs_plus_month3[i,3]
		stock_pairs_plus_1to3[i][4:(4+stock_pairs_plus_month1[i,2])]=stock_pairs_plus_month1[i][4:(4+stock_pairs_plus_month1[i,2])]
		stock_pairs_plus_1to3[i][(4+stock_pairs_plus_month1[i,2]):(4+stock_pairs_plus_month2[i,2]+stock_pairs_plus_month1[i,2])]=stock_pairs_plus_month2[i][4:(4+stock_pairs_plus_month2[i,2])]
		stock_pairs_plus_1to3[i][(4+stock_pairs_plus_month2[i,2]+stock_pairs_plus_month1[i,2]):(4+stock_pairs_plus_1to3[i,2])]=stock_pairs_plus_month3[i][4:(4+stock_pairs_plus_month3[i,2])]
	print "OK"


	with open("/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_plus_month1to3.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(stock_pairs_plus_1to3)):
			writer.writerow(stock_pairs_plus_1to3[i][:])

def together_month(together_dir,stock_number,trade_fee):
	print time.time()
	filedirs = os.listdir(together_dir)
	length_of_stock_number=len(stock_number)

	stock_pairs_plus_together = np.zeros((len(stock_number)*len(stock_number), 40), dtype=np.float32)

	temp_pairs=np.zeros((35),dtype=np.float32)
	for i in range(len(filedirs)):
		temp_stock_pairs_plus = []
		with open(together_dir+'/'+filedirs[i], "r") as csvfile:
			reader = csv.reader(csvfile)
			for item in reader:
				temp_stock_pairs_plus.append(item[0:35])
		for j in range(len(temp_stock_pairs_plus)):
			temp_pairs[0:35] = map(float32, temp_stock_pairs_plus[j][0:35])  ##从csv文件中取数
			for t in range(temp_pairs[2]):
				stock_pairs_plus_together[j][stock_pairs_plus_together[j][2]+4+t]=temp_pairs[4+t]
			stock_pairs_plus_together[j][2]+=temp_pairs[2]
			stock_pairs_plus_together[j][3]+=temp_pairs[3]
		del temp_stock_pairs_plus
		gc.collect()

	for i in range(len(stock_pairs_plus_together)):
		stock_pairs_plus_together[i][0]=np.floor(i / length_of_stock_number)
		stock_pairs_plus_together[i][1]=i % length_of_stock_number
		stock_pairs_plus_together[i][-3]=1
		for j in range(stock_pairs_plus_together[i][2]):
			stock_pairs_plus_together[i][-3]*=(stock_pairs_plus_together[i][4+j]-trade_fee)

	with open("/root/neural-networks-and-deep-learning/csv5minute/together_plus.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for i in range(len(stock_pairs_plus_together)):
			writer.writerow(stock_pairs_plus_together[i][:])
	print time.time()

def sort_together_plus(bingo_N):

	temp_stock_pairs_plus = []
	with open("/root/neural-networks-and-deep-learning/csv5minute/together_plus.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_pairs_plus.append(item[0:40])
	stock_pairs_plus = np.zeros((len(temp_stock_pairs_plus), 40), np.float32)
	for i in range(len(temp_stock_pairs_plus)):
		stock_pairs_plus[i][0:40] = map(float32, temp_stock_pairs_plus[i][0:40])  ##从csv文件中取数
		#stock_pairs_plus[i][34]=map(float32,temp_stock_pairs_plus[i][222])
	# del temp_stock_pairs_plus
	# gc.collect()
	temp_bingo_value = np.sort(stock_pairs_plus[:, 37])
	temp_bingo_index = np.argsort(stock_pairs_plus[:, 37])

	with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_plus_stock_pairs_mongth1to9.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in stock_pairs_plus[temp_bingo_index[int(-1*bingo_N):], :]:
			writer.writerow(row)



#	for t in range(int(stock_pa_plus[i][2])):
	# 		stock_pairs_plus[i][t+4]-=trade_fee
	# 		stock_pairs_plus[i][25]=1
	# for i in range(len(stock_pairs_plus)):
	# 	for t in range(int(stock_pairs_plus[i][2])):
	# 		stock_pairs_plus[i][25] *= stock_pairs_plus[i][t+4]
	#
	# temp_bingo_value = np.sort(stock_pairs_plus[:, 25])
	# temp_bingo_index = np.argsort(stock_pairs_plus[:, 25])
	#
	# with open("/root/neural-networks-and-deep-learning/csv5minute/binggo_plus_stock_pairs.csv", "w") as csvfile:
	# 	writer = csv.writer(csvfile)
	# 	for row in stock_pairs_plus[temp_bingo_index[int(-1*binggo_N):], :]:
	# 		writer.writerow(row)



inputfile_stock_number='/root/neural-networks-and-deep-learning/csv5minute/traindata/'
inputfile='/root/neural-networks-and-deep-learning/csv5minute/'
outputfile='/root/neural-networks-and-deep-learning/csv5minute/stock_pairs_plus_'
vertification_file='/root/neural-networks-and-deep-learning/csv5minute/month10/'
monthlist=['month1','month2','month3','month4','month5','month6','month7','month8','month9']
together_files='/root/neural-networks-and-deep-learning/csv5minute/stock_plus'



#get_base_G(inputfile_stock_number,1.06,0.85,1.005)

#stock_number = get_sorted_axle(inputfile_stock_number)
get_pairs(928,inputfile,0.003,180000)
# for i in range(len(monthlist)):
# 	temp_inputfile=inputfile+monthlist[i]+'/'
# 	temp_outputfile=outputfile+monthlist[i]+'.csv'
# 	get_base_G(temp_inputfile)
# 	get_eigenValue_G(1.05,1.05,1.02)
# 	get_pairs(len(stock_number),temp_outputfile)

# statistic_pairs_plus(80000,0.003)
# statistic_pairs_minus(5000,0.003)
# minus_result=vertification_test_minus(vertification_file,len(stock_number),1.02,1.02)
# plt.figure(1)
# plt.plot(minus_result[:,25])
# plt.show()
# print np.average(minus_result[:,25])


# plt.figure(2)
# plus_result=vertification_test_plus(vertification_file,len(stock_number),1.05,1.02)
# plt.plot(plus_result[:,25])
# plt.show()
# print np.average(plus_result[:,25])
# temp_sum=0
# for i in range(len(plus_result)):
# 	if plus_result[i][25]>1.0:
# 		temp_sum+=1

#together_month1to3_pairs()
#together_month(together_files,stock_number,0.003)
#sort_together_plus(80000)

print 'OK'






