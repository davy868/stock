#coding:utf-8
import os
import csv
import numpy as np
from multiprocessing import Pool
from dtw import dtw
import time
#from csv5read import get_file_name
"""
首先筛选一天中波动在2%到9%的股票，并保存到csv文件
其次计算有波动性的股票两两之间的dtw值，并保存到csv文件
"""
#filedirs = get_file_name('/root/neural-networks-and-deep-learning/month1')
# t1 = time.time()
# filedirs = os.listdir('/root/neural-networks-and-deep-learning/csv5minute/month1')
# for i in range(0, len(filedirs)):
#     filedirs[i] = '/root/neural-networks-and-deep-learning/csv5minute/month1/' + filedirs[i]

def get_day_prices_in_one_array(filename):
	#获得一天的股票分钟数据，返回一个数组，一行表示一个股票，一列表示一分钟，最后一列是股票代码
	day_prices_in_one_file = np.zeros([1000,244])
	temp_day_prices_in_one_file = []
	with open(filename) as f:
		reader = csv.reader(f)
		for row in reader:
			temp_day_prices_in_one_file.append(row[0:244])
		for i in range(0,len(temp_day_prices_in_one_file)):
			day_prices_in_one_file[i] = map(float,temp_day_prices_in_one_file[i])
	return day_prices_in_one_file

def choose_high_votality_stock(low_threshold,high_threshold,data_day_price):
	day_prices_high_votality =[]
	for i in range(0,len(data_day_price)):
		temp_min = np.min(data_day_price[i][0:243])
		temp_max = np.max(data_day_price[i][0:243])
		temp_mean = np.mean(data_day_price[i][0:243])
		if (temp_mean / temp_min > low_threshold) | (temp_max / temp_mean > low_threshold):
			day_prices_high_votality.append(data_day_price[i])

	with open("0109_high_voltality.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		for row in day_prices_high_votality:
			writer.writerow(row)
	temp_day_prices = np.zeros((len(day_prices_high_votality),244),dtype=np.float32)
	temp_day_prices = np.array(day_prices_high_votality)
	return temp_day_prices

def get_dtw_dist(x,y,ord):
	dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm([x - y], ord=ord))
	return dist
def z_score(mean,std,data):
	return (data - mean) / std

def get_pairs(t):
	if t == 0:
		step = 28
	elif t == 28:
		step = 34
	elif t == 62:
		step = 46
	else:
		step = 104

	day_price_dists = []
	for i in range(t, t+step):
		for j in range(i + 1, len(day_price_voltality)):
			day_price_dists.append(get_dtw_dist(day_price_voltality[i][0:243], day_price_voltality[j][0:243], 1))

	with open(str(t)+"_dist_data_choose0216.csv", "w") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(day_price_dists)

# day_prices_in_one_array = get_day_prices_in_one_array(filedirs[1])
#
# day_prices_in_one_array.astype(np.float32)
#
# day_price_voltality = choose_high_votality_stock(1.02,1.09,day_prices_in_one_array)
#
# for i in range(0,len(day_price_voltality)):
# 	day_price_voltality[i][0:243] = z_score(np.average(day_price_voltality[i][0:243]),np.std(day_price_voltality[i][0:243]),day_price_voltality[i][0:243])
#
# if  __name__ == "__main__":
# 	pool = Pool(processes=4)
# 	tasks = [0,28,62,108]
# 	pool.map(get_pairs,tasks)
#
# print time.time() - t1