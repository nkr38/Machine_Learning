# Student name(s): Noah Robinson
# Date modified: 8/15/23

import numpy as np
np.random.seed(0)

def calculateDistance(first, second):
    return np.sqrt(np.sum((first - second) ** 2))

def myKNN(Xtrain, Ytrain, XValid, k):
    myPredictions = []
    for sample in XValid:
        distancesToXtrain = [calculateDistance(sample, xTrain) for xTrain in Xtrain]
        nearestIndices = np.argsort(distancesToXtrain)[:k]
        nearestClasses = Ytrain[nearestIndices]
        unique, counts = np.unique(nearestClasses, return_counts=True)
        predictedClass = unique[np.argmax(counts)]
        myPredictions.append(predictedClass)
    return np.array(myPredictions)

csvData = np.genfromtxt('CTG.csv', delimiter=',', skip_header=2, usecols=range(1, 23))
xData = csvData[:, :-2]
yData = csvData[:, -1]

shuffledIndices = np.random.permutation(len(xData))
xShuffled = xData[shuffledIndices]
yShuffled = yData[shuffledIndices]

splitPoint = int(np.ceil(2/3 * len(xData)))
xTraining = xShuffled[:splitPoint]
yTraining = yShuffled[:splitPoint]
xValidation = xShuffled[splitPoint:]
yValidation = yShuffled[splitPoint:]

kValues = [1, 2, 3]
validationAccuracy = {}

for k in kValues:
    myPredictions = myKNN(xTraining, yTraining, xValidation, k)
    correctPredictions = np.sum(myPredictions == yValidation)
    accuracy = correctPredictions / len(yValidation)
    validationAccuracy[k] = accuracy

for k, accuracy in validationAccuracy.items():
    print(f"K:{k:2d}, Accuracy: {accuracy:.4f}")

bestK = max(validationAccuracy, key=validationAccuracy.get)
bestPredictions = myKNN(xTraining, yTraining, xValidation, bestK)
confusionMatrix = np.zeros((3, 3), dtype=int)

for i in range(len(yValidation)):
    trueClass = yValidation[i]
    predictedClass = bestPredictions[i]
    trueIndex = int(trueClass) - 1
    predictedIndex = int(predictedClass) - 1
    confusionMatrix[trueIndex, predictedIndex] += 1

print("\nConfusion Matrix:")
print(confusionMatrix)
