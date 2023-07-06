import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# yalefaces folder in current directory
folder_path = "./yalefaces"
file_list = os.listdir(folder_path)
data_matrix = np.zeros((154, 1600))

for i, file_name in enumerate(file_list):
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.isfile(file_path):
        image = Image.open(file_path)
        resized_image = image.resize((40, 40))
        flattened_image = np.array(resized_image).flatten()
        data_matrix[i, :] = flattened_image

# PCA
mean_centered_data = data_matrix - np.mean(data_matrix, axis=0)
covariance_matrix = np.cov(mean_centered_data.T)
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
indices = np.argsort(eigenvalues)[::-1]  # Descending order
top_eigenvectors = eigenvectors[:, indices[:2]]  # Top two eigenvectors
reduced_data = np.dot(mean_centered_data, top_eigenvectors)

# Convert imaginary numbers to real
if np.iscomplexobj(reduced_data):
    reduced_data = reduced_data.real

plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('Figure 1: 2D PCA Projection of data')
plt.show()

