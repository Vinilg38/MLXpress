import numpy as np
from MLXpress.math import pdf,hypothesis_test,distance_euclidean,linear_regression,covariance_matrix
import scipy.stats
x = 2
distribution = scipy.stats.norm(0, 1)
pdf(x, distribution)

data1 = [1, 2, 3, 4, 5]
data2 = [2, 4, 6, 8, 10]
hypothesis_test(data1, data2)

x1 = np.array([1, 2])
x2 = np.array([3, 4])
distance_euclidean(x1, x2)

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])
linear_regression(x, y)

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
covariance_matrix(matrix)