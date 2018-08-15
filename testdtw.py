import os
import csv
import numpy as np
from multiprocessing import Pool
import time

filename0 = '/root/neural-networks-and-deep-learning/csv5minute/0_dist_data.csv'
filename28 = '/root/neural-networks-and-deep-learning/csv5minute/28_dist_data.csv'
filename62 = '/root/neural-networks-and-deep-learning/csv5minute/62_dist_data.csv'
filename108 = '/root/neural-networks-and-deep-learning/csv5minute/108_dist_data.csv'
header_row = []
header_row =np.array(header_row)
with open(filename0) as f:
	reader0 = csv.reader(f)
	header_row = next(reader0)
print len(header_row)
with open(filename28) as f28:
	reader28 = csv.reader(f28)
	header_row = np.append(header_row,next(reader28))
print len(header_row)
with open(filename62) as f62:
	reader62 = csv.reader(f62)
	header_row = np.append(header_row,next(reader62))
print len(header_row)
with open(filename108) as f108:
	reader108 = csv.reader(f108)
	header_row = np.append(header_row,next(reader108))

rowofdata = header_row.T

data_dist = zip(range(len(rowofdata)),rowofdata)
data_dist = sorted(data_dist,key=lambda x:x[1])

with open("sorted_original_dist_data.csv", "w") as csvfile:
	writer = csv.writer(csvfile)
	for row in data_dist:
		writer.writerow(row)

print len(header_row)
print "ok"