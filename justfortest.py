#coding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import re
import time
import itchat
from itchat.content import *
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import operator
import csv
import math
from matplotlib import pyplot
import gc
from sqlalchemy import create_engine
import tushare as ts
import pymysql
import pandas as pd


stock_number = []
with open("/root/neural-networks-and-deep-learning/csv5minute/stock_number.csv", "r") as csvfile:
	reader = csv.reader(csvfile)
	for item in reader:
		stock_number = item




temp=0
for i in range(0,len(stock_number)):
	df = ts.get_realtime_quotes(stock_number[i])
	for t in range(df['pre_close'].size):
		if (float(df['high'][t])+0.00001)/(float(df['pre_close'][t])+0.00001) > 1.005 and  (float(df['high'][t])+0.00001)/(float(df['pre_close'][t])+0.00001) < 1.109 :
			temp+=1

print temp
print 'OK'
