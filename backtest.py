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

def backtest(startX,startY):
	times_for_buy=200
	num_of_pairs=2500
	num_of_stocks=4000
	stocks_for_buy_month2=np.zeros((times_for_buy,num_of_stocks,4))
	times_for_oneday_buy=6
	time_between_buy=30

	stock_pairs_for_buy=np.zeros((times_for_buy,num_of_pairs,6),dtype=np.float32)

	index_of_pairs=0

	with open("stock_number_month1.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			temp_stock_number=item

	stock_number=np.zeros((len(temp_stock_number)),dtype=np.float32)
	for i in range(len(temp_stock_number)):
		stock_number[i]=np.float32(temp_stock_number[i])

	with open("stock_pairs_mark_month1.csv", "r") as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			if item[3]<=item[4]:
				stock_pairs_for_buy[:,index_of_pairs,0]=stock_number[int(float(item[0]))]-600000
				stock_pairs_for_buy[:,index_of_pairs,1]=stock_number[int(float(item[1]))]-600000
				index_of_pairs+=1
			else :
				stock_pairs_for_buy[:, index_of_pairs, 0] = stock_number[int(float(item[1]))] - 600000
				stock_pairs_for_buy[:, index_of_pairs, 1] = stock_number[int(float(item[0]))] - 600000
				index_of_pairs += 1

	stock_prices_day_month2=np.zeros((num_of_stocks,244),dtype=np.float32)
	end_of_everytime_buy=np.zeros(times_for_buy,dtype=np.float32)

	old_buyed_stock=np.zeros((1000),dtype=np.float32)
	new_buying_stock=np.zeros((1000),dtype=np.float32)


	tradesum=0
	filedirs = os.listdir('/root/neural-networks-and-deep-learning/csv5minute/month2')
	for i in range(0, len(filedirs)):
		filedirs[i] = '/root/neural-networks-and-deep-learning/csv5minute/month2/' + filedirs[i]
		day_prices_in_one_array = get_day_prices_in_one_array(filedirs[i])
		day_prices_in_one_array.astype(np.float32)

		for j in range(len(day_prices_in_one_array)):
			if day_prices_in_one_array[j,243]>600000:
				stock_prices_day_month2[int(day_prices_in_one_array[j,243]-600000),:]=day_prices_in_one_array[j,:]

		for u in range(times_for_oneday_buy):
			for t in range(startX,startY):
				tempAnew=stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy*i+u,t,0]),40+u*time_between_buy]
				tempAnewer = stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 0]), 40 + (u+1) * time_between_buy]
				tempAold = stock_prices_day_month2[int(stock_pairs_for_buy[ times_for_oneday_buy* i + u, t, 0] ), 10 + u * time_between_buy]
				tempBnew = stock_prices_day_month2[int(stock_pairs_for_buy[ times_for_oneday_buy* i + u, t, 1] ), 40 + u * time_between_buy]
				tempBnewer = stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 1] ), 40 + (u + 1) * time_between_buy]
				tempBold = stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 1] ), 10 + u * time_between_buy]

				rateA=tempAnew/tempAold
				rateB=tempBnew/tempBold
				if rateA>1.02 :
					if abs(rateA-rateB)>0.02:
						if rateA>rateB:
							#stock_pairs_for_buy[7 * i + u, t, 3] = 1
							stocks_for_buy_month2[ times_for_oneday_buy* i + u,int(stock_pairs_for_buy[ times_for_oneday_buy* i + u, t, 1]),0]+=1
							stocks_for_buy_month2[times_for_oneday_buy * i + u, int(stock_pairs_for_buy[times_for_oneday_buy* i + u, t, 1]), 1] =tempBnew
							stocks_for_buy_month2[times_for_oneday_buy * i + u, int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 1]), 3] = tempBnewer
							plt.plot(z_score(np.average(stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy*i+u,t,0])][0:243]), np.std(stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy*i+u,t,0])][0:243]),
							        stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 0])][0:243]))

							plt.plot(z_score(np.average(
								stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 1])][
								0:243]), np.std(
								stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 1])][
								0:243]),
							                 stock_prices_day_month2[
								                 int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 1])][0:243]))
							#plt.plot(stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy*i+u,t,0])][0:243])
							#plt.plot(stock_prices_day_month2[int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 1])][0:243])
							plt.show()
						else:
							stocks_for_buy_month2[times_for_oneday_buy * i + u, int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 0]), 0] += 1
							stocks_for_buy_month2[times_for_oneday_buy * i + u, int(stock_pairs_for_buy[ times_for_oneday_buy* i + u, t, 0]), 1] = tempAnew
							stocks_for_buy_month2[times_for_oneday_buy * i + u, int(stock_pairs_for_buy[times_for_oneday_buy * i + u, t, 0]), 3] = tempAnewer

			tempsum=np.sum(stocks_for_buy_month2[times_for_oneday_buy * i + u,:,0])
			tradesum+=tempsum

			for v in range(num_of_stocks):
				if stocks_for_buy_month2[times_for_oneday_buy * i + u, v, 1]!=0:
					stocks_for_buy_month2[times_for_oneday_buy * i + u, v, 2]=(stocks_for_buy_month2[times_for_oneday_buy * i + u,v,0]/tempsum)/stocks_for_buy_month2[times_for_oneday_buy * i + u, v, 1]
				end_of_everytime_buy[times_for_oneday_buy * i + u] +=stocks_for_buy_month2[times_for_oneday_buy * i + u, v, 2]*stocks_for_buy_month2[times_for_oneday_buy * i + u, v, 3]

	g=1.0

	for r in range(90):
		if end_of_everytime_buy[r]==0:
			end_of_everytime_buy[r]=1
		g=g*end_of_everytime_buy[r]

	#print "OK"
	return g,tradesum/60












