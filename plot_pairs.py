import matplotlib.pyplot as plt
import csv
import numpy as np
from dtw import dtw

def get_dtw_dist(x,y,ord):
	dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm([x - y], ord=ord))
	return dist

filename='/root/neural-networks-and-deep-learning/csv5minute/sh_20140130_minute_data_dist_data.csv'
pairs1 = np.zeros(243)
pairs2 = np.zeros(243)
with open(filename) as f:
	reader = csv.reader(f)
	for i,row in enumerate(reader):
		if i == 28:
			pairs1 = map(float,row[0:243])
		elif i == 126:
			pairs2 = map(float,row[0:243])

pairs1 = (pairs1 - np.average(pairs1)) / np.std(pairs1)
pairs2 = (pairs2 - np.average(pairs2)) / np.std(pairs2)

dist = get_dtw_dist(pairs1,pairs2,1)

print dist
plt.plot(pairs1)
plt.plot(pairs2)
plt.show()

print 'OK'
