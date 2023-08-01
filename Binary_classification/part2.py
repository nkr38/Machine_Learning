# Name of student: Noah Robinson
# Date modified: 7/29/23

import numpy as np
import pandas as pd

# Seed the random number generate with zero prior to randomizing the data
np.random.seed(0)

# Reads in the data
csvData = pd.read_csv('spambase.data', header=None)

xInput = csvData.iloc[:, :-1].values
yInput = csvData.iloc[:, -1].values

# Randomize and split data
randomOrder = np.random.permutation(len(csvData))
xRandom = xInput[randomOrder]
yRandom = yInput[randomOrder]

numTrain = int(np.ceil( 2 / 3 * len(csvData)))
xTraining = xRandom[:numTrain]
yTraining = yRandom[:numTrain]
xValidation = xRandom[numTrain:]
yValidation = yRandom[numTrain:]

trainingMean = np.mean(xTraining, axis=0)
trainingStd = np.std(xTraining, axis=0)

# Standardizes (z-scores) the data (except for the last column of course) using the training data
xTrainingStd = (xTraining - trainingMean) / trainingStd
xValidationStd = (xValidation - trainingMean) / trainingStd

# Calc optimal direction of projection for LDA
firstClassMean = np.mean(xTrainingStd[yTraining == 0], axis=0)
secondClassMean = np.mean(xTrainingStd[yTraining == 1], axis=0)
projection_direction = secondClassMean - firstClassMean
X_val_projected = np.dot(xValidationStd, projection_direction)
X_train_projected = np.dot(xTrainingStd, projection_direction)

# Classifies each validation sample by projecting it and assinging it the class whose training post-projection mean it is closes to
yValPredicted = np.zeros(len(yValidation))
for i in range(len(yValidation)):
    firstClass = np.linalg.norm(X_val_projected[i] - firstClassMean)
    secondClass = np.linalg.norm(X_val_projected[i] - secondClassMean)
    if firstClass < secondClass:
        yValPredicted[i] = 0 
    else:
        yValPredicted[i] = 1

# Classify training samples
yTrainPredicted = np.zeros(len(yTraining))
for i in range(len(yTraining)):
    firstClass = np.linalg.norm(X_train_projected[i] - firstClassMean)
    secondClass = np.linalg.norm(X_train_projected[i] - secondClassMean)
    if firstClass < secondClass:
        yTrainPredicted[i] = 0 
    else:
        yTrainPredicted[i] = 1

# "Accuracy of the training and validation data" - This was vague for me, I feel like we should just calculate the validation 
# accuracy but I'm also calculating training accuracy just to be safe
trainAccuracy = np.sum(yTraining == yTrainPredicted) / len(yTraining)
valAccuracy = np.sum(yValidation == yValPredicted) / len(yValidation)

# Precision, recall, and f-measure for the validation set
trueP = np.sum((yValidation == 1) & (yValPredicted == 1))
falseP = np.sum((yValidation == 0) & (yValPredicted == 1))
falseN = np.sum((yValidation == 1) & (yValPredicted == 0))
valPrecision = trueP / (trueP + falseP)
valRecall = trueP / (trueP + falseN)
fMeasure = 2 * (valPrecision * valRecall) / (valPrecision + valRecall)

print(f"Training Accuracy:    {trainAccuracy:.4f}")
print(f"Validation Accuracy:  {valAccuracy:.4f}")
print(f"Precision:            {valPrecision:.4f}")
print(f"Recall:               {valRecall:.4f}")
print(f"F-measure:            {fMeasure:.4f}")