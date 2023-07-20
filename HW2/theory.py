# Name of student: Noah Robinson
# Date modified: 7/16/23

import numpy as np

xValues = np.array([-2, -5, -3, 0, -8, -2, 1, 5, -1, 6]) 
yValues = np.array([1, -4, 1, 3, 11, 5, 0, -1, -3, 1])

xBiased = np.column_stack((np.ones(len(xValues)), xValues))
wMatrix = np.linalg.inv(xBiased.T.dot(xBiased)).dot(xBiased.T).dot(yValues)
print("Coefficients:", wMatrix)

yPrediction = wMatrix[1] * xValues + wMatrix[0]

rmseValue = np.sqrt(np.mean((yValues - yPrediction) ** 2))

# According to formula in slides
smapeValue = np.mean(np.abs(yValues- yPrediction) / ((np.abs(yPrediction) + np.abs(yValues))))

print("RMSE:", rmseValue)
print("SMAPE:", smapeValue)