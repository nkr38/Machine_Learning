# Name of student: Noah Robinson
# Date modified: 7/31/23

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
terminationThreshold = 1e-6
learningRate = 0.1
numEpochs = 1000

# Seed the random number generate with zero prior to randomizing the data
np.random.seed(0)

# Reads in the data
csvData = pd.read_csv('spambase.data', header=None)

# Separate features (X) and class labels (y)
xInput = csvData.iloc[:, :-1].values
yInput = csvData.iloc[:, -1].values

# Randomizes the data
randomOrder = np.random.permutation(len(csvData))
xRandom = xInput[randomOrder]
yRandom = yInput[randomOrder]

# Selects the first 2/3 (round up) of the data for training and the remaining for validation.
numTrain = int(np.ceil( 2 / 3 * len(csvData)))
xTraining = xRandom[:numTrain]
yTraining = yRandom[:numTrain]
xValidation = xRandom[numTrain:]
yValidation = yRandom[numTrain:]

# Standardizes (z-scores) the data (except for the last column of course) using the training data
trainingMean = np.mean(xTraining, axis=0)
trainingStd = np.std(xTraining, axis=0)
xTrainingStd = (xTraining - trainingMean) / trainingStd
xValidationStd = (xValidation - trainingMean) / trainingStd

# Add a column of ones to xTrainingStd for the bias term
xTrainingStd = np.hstack((np.ones((xTrainingStd.shape[0], 1)), xTrainingStd))

# Initialize the logistic regression parameters with an additional dimension for bias term
numFeatures = xTrainingStd.shape[1]
theta = np.random.rand(numFeatures, 1)

logLossTrain = []
logLossVal = []

# Sigmoid formula
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

# Helper function to compute log loss
def logLoss(yTrue, yPred):
    epsilon = 1e-15
    yPred = np.clip(yPred, epsilon, 1 - epsilon) # To avoid log(0) issues
    return -np.mean(yTrue * np.log(yPred) + (1 - yTrue) * np.log(1 - yPred))

# Trains a logistic classifier, keeping track of the log loss for both the training and 
# validation data as you train.
for epoch in range(numEpochs):
    yPredTrain = sigmoid(np.dot(xTrainingStd, theta))
    lossTrain = logLoss(yTraining.reshape((-1, 1)), yPredTrain)
    logLossTrain.append(lossTrain)

    xValidationStdBias = np.hstack((np.ones((xValidationStd.shape[0], 1)), xValidationStd))
    yPredVal = sigmoid(np.dot(xValidationStdBias, theta))
    lossVal = logLoss(yValidation.reshape((-1, 1)), yPredVal)
    logLossVal.append(lossVal)
    
    # Computes the gradient of the log loss
    gradient = np.dot(xTrainingStd.T, (yPredTrain - yTraining.reshape((-1, 1)))) / len(yTraining)
    theta -= learningRate * gradient

    if epoch > 0 and abs(logLossTrain[-1] - logLossTrain[-2]) < terminationThreshold:
        break

# Add a column of ones for bias
xValidationStd = np.hstack((np.ones((xValidationStd.shape[0], 1)), xValidationStd))

# Classify each validation sample using your trained model, choosing an observation to be spam
# if the output of the model is ≥ 50%.
yValPredicted = (sigmoid(np.dot(xValidationStd, theta)) >= 0.5).astype(int)

# Accuracy of the training and validation data and the precision, recall, and fmeasure for the validation set
trueP = 0
trueN = 0
falseP = 0
falseN = 0

for i in range(len(yValPredicted)):
    if yValidation[i] == 1 and yValPredicted[i] == 1:
        trueP = trueP + 1
    elif yValidation[i] == 0 and yValPredicted[i] == 0:
        trueN = trueN + 1
    if yValidation[i] == 0 and yValPredicted[i] == 1:
        falseP = falseP + 1
    elif yValidation[i] == 1 and yValPredicted[i] == 0:
        falseN = falseN + 1

valPrecision = trueP / (trueP + falseP)
valRecall = trueP / (trueP + falseN)
fMeasure = 2 * (valPrecision * valRecall) / (valPrecision + valRecall)
valAccuracy = (trueP + trueN) / len(yValidation)

print(f"Precision:   {valPrecision:.4f}")
print(f"Recall:      {valRecall:.4f}")
print(f"F-measure:   {fMeasure:.4f}")
print(f"Accuracy:    {valAccuracy:.4f}")

plt.plot(range(len(logLossTrain)), logLossTrain, label='Training Data')
plt.plot(range(len(logLossVal)), logLossVal, label='Validation Data')
plt.xlabel('Epoch')
plt.ylabel('Log-Loss')
plt.legend()
plt.show()