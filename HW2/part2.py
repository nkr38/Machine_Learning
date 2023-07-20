# Name of student: Noah Robinson
# Date modified: 7/16/23

import pandas as pd
import numpy as np

np.random.seed(0)
csvData = pd.read_csv('insurance.csv')
csvData = csvData.sample(frac=1).reset_index(drop=True) # Randomize

numTrain = int((2/3) * len(csvData))

trainData = csvData[:numTrain]
valData = csvData[numTrain:]

featSelect = ['sex', 'smoker', 'region']
trainData = pd.get_dummies(trainData, columns=featSelect)
validationData = pd.get_dummies(valData, columns=featSelect)

xTraining = trainData.drop('charges', axis=1).values
yTraining = trainData['charges'].values

xValidation = validationData.drop('charges', axis=1).values
yValidation = validationData['charges'].values

def computeCoefficients(X, y):
    X = np.insert(X.T, 0, 1, axis=0) # Bias
    # W = ((X^T*X)^-1)*X^T*y
    X_cross = np.dot(np.linalg.pinv(np.dot(X, X.T)), X)
    w = np.dot(X_cross, y)
    return w

W = computeCoefficients(xTraining, yTraining)

def predictMatrix(X, w):
    X = np.insert(X, 0, 1, axis=1)
    return np.dot(X, w)

predictValidation = predictMatrix(xValidation, W)
predictTraining = predictMatrix(xTraining, W)

def rmse(actual, pred):
    return np.sqrt(np.mean((actual - pred) ** 2))

def smape(actual, pred):
    return np.mean(np.abs(actual- pred) / ((np.abs(pred) + np.abs(actual))))

# RMSE & SMAPE
print("Validation RMSE:", rmse(yValidation, predictValidation))
print("Validation SMAPE:", smape(yValidation, predictValidation))
print("Training RMSE:", rmse(yTraining, predictTraining))
print("Training SMAPE:", smape(yTraining, predictTraining))