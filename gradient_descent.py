# file: gradient_descent.py
# -------------------------
# Finds the local minimum of the cost function using gradient descent

import numpy as np
from numpy import *
from common_functions import *
import matplotlib.pyplot as plt
from random import randrange

def designer (data, degree):
	""" Returns the design matrix of the data. """

	xs = data[:, 0]
	xs = feature_scaling(xs) # returns normalized x values
	power_list = []
	design = ones(degree + 1)

	# Creating design matrix with powers of the x values
	for x in xs:
		for deg in range(0, degree+1):
			power_list.append(x ** deg)
		design = vstack((design, power_list))
		power_list = []

	design = design[1:]

	return design

def hypothesis (thetas, design):
	""" hypothesis function: the dot product 
	of the data array and theta array. """
	
	try:
		return np.dot(design, thetas)
	except ValueError:
		return np.dot(thetas, design)

def cost_function (thetas, design, data):
	""" Cost function without regularization. """
	
	n = data.shape[1]
	ys = data[:, n-1]
	m = 2 * len(data) # divisor 

	cost = hypothesis(thetas, design) - ys
	
	return (1. / m) * dot(cost.transpose(), cost)

def cost_function_r (thetas, design, data, reg):
	""" cost function with regularization.
	reg is the regularization parameter. """
	
	m, n = data.shape
	ys = data[:, n-1]
	div = (1. / len(data)) # divisor 

	difference = hypothesis(thetas, design) - ys
	
	return div * dot(difference.transpose(), difference) + reg * dot(thetas.transpose(), thetas)

def gradient_descent (thetas, design, data):
	""" Returns modified weights by finding the minimum of the cost function """

	n = data.shape[1] # number of columns
	ys = data[:, n-1]

	m = len(data) # divisor 
	alpha = .01 # learning rate

	for i in range(100000):
		for t in range(len(thetas)):
			gradient = (1. / m) * dot((ys - hypothesis(thetas, design)), -design[:, t])
			thetas[t] = thetas[t] - alpha * gradient

	return thetas

def gradient_descent_r (thetas, design, data, reg):
	""" Gradient descent with regularization parameter.
	Returns modified weights by finding the minimum of the cost function. """

	n = data.shape[1] # number of columns
	ys = data[:, n-1]

	m = len(data) # divisor 
	alpha = .01 # learning rate

	for i in range(100000):
		for t in range(len(thetas)):
			gradient = dot((ys - hypothesis(thetas, design)), -design[:, t])
			thetas[t] = thetas[t] - (gradient + (reg * thetas[t])) * alpha * (1. / m)

	return thetas

def run_graph (design, thetas, data):
	""" Plots the hypothesis values and actual values of the data. """
	
	xs = data[:, 0]
	xs = feature_scaling(xs) # returns normalized x values
	ys = data[:, 1] # y values

	plt.figure()
	plt.plot(xs, hypothesis(thetas, design), 'b') # hypothesis plot
	plt.plot(xs, ys, 'ro') # actual plot
	plt.xlabel('U')
	plt.ylabel('Time')
	plt.show()

if __name__ == "__main__":

	data = genfromtxt("u_time_tree.csv", dtype=float, delimiter=',') # data array
	design = designer(data, degree=2) # design matrix
	n = design.shape[1] # number of columns in design matrix
	thetas = random.rand(n)
	thetas = gradient_descent(thetas, design, data)
	print (thetas)

	run_graph(design, thetas, data)
