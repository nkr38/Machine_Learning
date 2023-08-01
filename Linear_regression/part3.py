# Name of student: Noah Robinson
# Date modified: 7/19/23

import pandas as pd
import numpy as np

csvData = pd.read_csv('insurance.csv')

featSelect = ['sex', 'smoker', 'region']
processedData = pd.get_dummies(csvData, columns=featSelect)

xInput = processedData.drop('charges', axis=1).values
yInput = processedData['charges'].values

numFolds = 223 # Edit for foldz
rmseArray = []

for run in range(20):
    np.random.seed(run)
    randomOrder = np.random.permutation(len(processedData))
    xRandom = xInput[randomOrder]
    yRandom = yInput[randomOrder]

    chunkSize = len(processedData) // numFolds
    squaredErrorsArray = []
    
    for i in range(numFolds):
        chunkStart = i * chunkSize
        chunkEnd = chunkStart + chunkSize
        
        xValidation = xRandom[chunkStart:chunkEnd]
        yValidation = yRandom[chunkStart:chunkEnd]
        
        xTraining = np.concatenate([xRandom[:chunkStart], xRandom[chunkEnd:]])
        yTraining = np.concatenate([yRandom[:chunkStart], yRandom[chunkEnd:]])
        
        xTraining = np.insert(xTraining, 0, 1, axis=1)
        xValidation = np.insert(xValidation, 0, 1, axis=1)

        xTransposeXInverted = np.linalg.pinv(np.dot(xTraining.T, xTraining))
        xTransposeY = np.dot(xTraining.T, yTraining)
        wMatrix = np.dot(xTransposeXInverted, xTransposeY)
        
        yPredict = np.dot(xValidation, wMatrix)
        
        squaredError = (yValidation - yPredict) ** 2
        squaredErrorsArray.append(squaredError)
    
    rmseValue = np.sqrt(np.mean(squaredErrorsArray))
    rmseArray.append(rmseValue)

print("S =", numFolds)
print("Mean RMSE =", np.mean(rmseArray))
print("Std. RMSE =", np.std(rmseArray))
