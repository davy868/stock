#coding:utf-8
import os
import csv
import numpy as np
from multiprocessing import Pool
import time
"""
用于将每个月的每只股票的tick数据转变成标准格式的分钟数据，一天一个表格
"""
def get_times():
    filename = '/root/PycharmProjects/stock_pairs/data2/sh201401d/sh_20140102/600000_20140102.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        times = []
        prices_intime=[]
        for row in reader:
            times.append(row[2][11:17])
            #prices_intime.append(row[2:4])

    i = 0
    while i <= len(times)-2:
        if times[i+1] == times[i] :
            times.pop(i+1)
        else:
            i += 1
    times.insert(242,'15:00:')
    return times

def get_day_prices_one_file(times,filename):

    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
        prices_intime=[]
        for row in reader:
            prices_intime.append(row[2:4])

    prices_intime.insert(len(prices_intime),['end','end'])

    j = 0
    temp_intime_prices = np.array([],dtype=float)
    day_prices = np.array([],dtype=float)


################################################
#以下代码段为了解决交易数据不从09:25开始的问题
    while '09:25:' not in prices_intime[j][0]:
        j=j+1
        if j>len(prices_intime)-1:
            break
    if j>(len(prices_intime)-1):
        j=0
        while '09:30:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
            j=0
            while '09:31:' not in prices_intime[j][0]:
                j=j+1
                if j>len(prices_intime)-1:
                    break
    if j>len(prices_intime)-1:
        j=0
        while '09:32:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        while '09:33:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        while '09:34:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        while '09:35:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        while '09:36:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        while '09:37:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        while '09:38:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        while '09:39:' not in prices_intime[j][0]:
            j=j+1
            if j>len(prices_intime)-1:
                break
    if j>len(prices_intime)-1:
        j=0
        print "trade data error"#假设交易数据至少从09:39开始
##########################################





    for i in range(0,len(times)):
        while times[i] in prices_intime[j][0]:
            temp_intime_prices = np.append(temp_intime_prices,float(prices_intime[j][1]))
            j += 1

        if temp_intime_prices.size == 0:
            if i == 0:
                day_prices = np.append(day_prices,0.0)
            else:
                day_prices = np.append(day_prices,day_prices[i-1])
        else:
            day_prices = np.append(day_prices,temp_intime_prices.mean())
            temp_intime_prices = np.array([])

    for i in range(len(times)-1,-1,-1):
        if day_prices[i] == 0.0 :
            day_prices[i] = day_prices[i+1]

    day_prices = np.append(day_prices,filename[-19:-13])

    return day_prices



def get_file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                L.append(os.path.join(root, file))
    return L

def get_day_price_one_day(filedir):
    times = get_times()
    files = get_file_name(filedir)
    with open(filedir[-11:] + "_minute_data.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for file in files:
            writer.writerow(get_day_prices_one_file(times, file))

def caculate_day_prices(filedirs):
    for i in range(0,len(filedirs)):
        get_day_price_one_day(filedirs[i])

t1 = time.time()
filedirs = os.listdir('/root/PycharmProjects/stock_pairs/data2/sh201511d')
for i in range(0, len(filedirs)):
    filedirs[i] = '/root/PycharmProjects/stock_pairs/data2/sh201511d/' + filedirs[i]

if  __name__ == "__main__":
    pool = Pool(processes=4)

    pool.map(caculate_day_prices,[filedirs[0:4],filedirs[4:8],filedirs[8:12],filedirs[12:len(filedirs)]])

#caculate_day_prices(filedirs[12:len(filedirs)])

print time.time() - t1

print "Ok"
