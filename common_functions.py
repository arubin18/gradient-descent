# file: common_functions.py
# -------------------------
# feature scaling normalizes data

import numpy as np

def mean (xs):
	""" mean of x values """
	x_mean = sum(xs) / len(xs) # average for x values
	return x_mean

def ranger (xs):
	""" range of x values """
	r = max(xs) - min(xs)
	return r

def feature_scaling (xs):
	""" feature scaling """
	norms = []
	for x in xs:
		n = (x - mean(xs)) / ranger(xs)  # apply normalization 
		norms.append(n) # append normalization
	norms = np.array(norms)

	return norms