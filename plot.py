# file: plot.py
# -------------
# gradient descent on housing data

from numpy import *
from gradient_descent import *
from random import randrange
import matplotlib.pyplot as plt

data = genfromtxt("housing_data.txt", dtype=float, delimiter=',') # data array
design = designer(data, degree=2) # design matrix
n = design.shape[1] # number of columns in design matrix
thetas = random.rand(n)
thetas = gradient_descent(thetas, design, data)
print (thetas)

run_graph(design, thetas, data)