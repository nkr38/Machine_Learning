import os
from PIL import Image
import numpy as np
import cv2

# yalefaces folder in current directory
folder_path = "./yalefaces"
file_list = os.listdir(folder_path)
data_matrix = np.zeros((len(file_list), 1600))

# Add files to the list, then flatten and add to the matrix
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

# Giving me warnings if I dont have this
if np.iscomplexobj(mean_centered_data):
    mean_centered_data = mean_centered_data.real

# Find the index of our image
image_index = file_list.index("subject02.centerlight")

# Video params
output_file = 'reconstructed_video.mp4'
frame_rate = 10
frame_size = (40, 40)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

# Reverse PCA with more and more of the best eigenvectors
for k in range(1, len(eigenvalues) + 1):
    top_eigenvectors = eigenvectors[:, :k] # Top k eigenvectors
    reduced_data = np.dot(mean_centered_data, top_eigenvectors)
    reconstructed_data = np.dot(reduced_data, top_eigenvectors.T) + np.mean(data_matrix, axis=0)
    reconstructed_data = reconstructed_data.real # Again, warnings if I dont have this
    reconstructed_image = reconstructed_data[image_index].reshape((40, 40)).astype(np.uint8)
    # Add image to video
    reconstructed_image_bgr = cv2.cvtColor(reconstructed_image, cv2.COLOR_GRAY2BGR) # Video wasnt working without this
    video.write(reconstructed_image_bgr)

video.release()