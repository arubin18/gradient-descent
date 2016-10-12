from gradient_descent import *

data = genfromtxt("housing_data.txt", dtype=float, delimiter=',') # data array
design = designer(data, degree=2) # design matrix
n = design.shape[1] # number of columns in design matrix
thetas = random.rand(n)
thetas = gradient_descent(thetas, design, data)
print ("THETAS: " + str(thetas))

run_graph(design, thetas, data)