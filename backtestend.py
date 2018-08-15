from backtest import *
import numpy as np
import matplotlib.pyplot as plt

print backtest(2300,2500)



step=100;
getOffit=np.zeros(1,dtype=np.float32)
result=np.zeros((50,2),dtype=np.float32)
for i in range(0,2401,step):
	result[int(i/step)][:]=backtest(i,i+step)
	print backtest(i,i+step)
	print i,i+step

plt.plot(result[:,0])
plt.show()
print "OK"