import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 

Xorig = np.array([[0,1], [0,0], [1,1], [0,0], [1,1], [1,0], [1,0], [1,1], [2,0], [2,1]])
X = np.array([[0,1], [0,0], [1,1], [0,0], [1,1], [1,0], [1,0], [1,1], [2,0], [2,1]])
n, m = X.shape

meanPoint = X.mean(axis = 0)
# subtract mean point
X = X - meanPoint
# Compute covariance matrix
C = np.dot(X.T, X) / (n-1)
# Eigen decomposition
eigen_vals, eigen_vecs = np.linalg.eig(C)
# Project X onto PC space
#X_pca = np.dot(X, eigen_vecs)

mean_centered_data = X - np.mean(X, axis=0)
covariance_matrix = np.cov(mean_centered_data.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
X_pca = np.dot(mean_centered_data, eigenvectors)

plt.scatter(Xorig[:,0],Xorig[:,1])
plt.scatter(X_pca[:,0],X_pca[:,1])

print("Eigen Vals:")
print(eigen_vals)
print(eigen_vecs)
print(preprocessing.normalize(eigen_vecs))

print(X_pca)

plt.show()
