import numpy as np
from  scipy.optimize import minimize

def objectivefun(x):
	return x[0] + x[1]

def constraint(x):
	return -0.5*x[0]**2+213.5*x[0]+x[1]-214-Q
def constraint_integer(x):
	return max([x[i] - int(x[i]) for i in range(len(x))])

Q = 6729
x0 = np.zeros(2)
x0[0] = 1
x0[1] = 1


b = (1,214)
bnds = (b,b)

con1 = {'type':'eq','fun':constraint}
con2 = {'type':'eq','fun':constraint_integer}
cons = ([con1])

solution = minimize(objectivefun,x0,method='SLSQP',bounds=bnds,constraints=cons)

x = solution.x

print x

print 'OK'