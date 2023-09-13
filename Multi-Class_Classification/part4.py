# Student name(s): Noah Robinson
# Date modified: 8/17/23

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

np.random.seed(0)

class TreeNode:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.left = None
        self.right = None

def calculateDistance(first, second):
    return np.sqrt(np.sum((first - second) ** 2))

def myKNN(xTrain, yTrain, xValid, k):
    myPredictions = []
    for sample in xValid:
        distancesToXTrain = [calculateDistance(sample, xTr) for xTr in xTrain]
        nearestIndices = np.argsort(distancesToXTrain)[:k]
        nearestClasses = yTrain[nearestIndices]
        predictedClass = np.bincount(nearestClasses).argmax()
        myPredictions.append(predictedClass)
    return np.array(myPredictions)

def findEntropy(labels):
    probabilities = np.bincount(labels) / len(labels)
    entropyValue = -np.sum(probabilities * np.log2(probabilities + 1e-8))
    return entropyValue

def findBestSplit(data, labels):
    bestEntropy = float('inf')
    bestFeature = None
    bestThreshold = None
    for feature in range(data.shape[1]):
        uniqueValues = np.unique(data[:, feature])
        for threshold in uniqueValues:
            leftLabels = labels[data[:, feature] <= threshold]
            rightLabels = labels[data[:, feature] > threshold]
            entropyLeft = findEntropy(leftLabels)
            entropyRight = findEntropy(rightLabels)
            weightedEntropy = (len(leftLabels) / len(labels)) * entropyLeft + (len(rightLabels) / len(labels)) * entropyRight
            if weightedEntropy < bestEntropy:
                bestEntropy = weightedEntropy
                bestFeature = feature
                bestThreshold = threshold

    return bestFeature, bestThreshold

def createDecisionTree(data, labels, depth=0, maxDepth=None):
    if maxDepth is not None and depth >= maxDepth:
        return TreeNode(None, np.bincount(labels).argmax())

    if np.all(labels == labels[0]):
        return TreeNode(None, labels[0])

    if data.shape[1] == 0:
        return TreeNode(None, np.bincount(labels).argmax())

    bestFeature, bestThreshold = findBestSplit(data, labels)
    leftIndices = data[:, bestFeature] <= bestThreshold
    rightIndices = data[:, bestFeature] > bestThreshold

    leftTree = createDecisionTree(data[leftIndices], labels[leftIndices], depth + 1, maxDepth)
    rightTree = createDecisionTree(data[rightIndices], labels[rightIndices], depth + 1, maxDepth)

    node = TreeNode((bestFeature, bestThreshold), None)
    node.left = leftTree
    node.right = rightTree
    return node

def predictTree(node, sample):
    while node.left is not None and node.right is not None:
        feature, threshold = node.data
        if sample[feature] <= threshold:
            node = node.left
        else:
            node = node.right
    return node.label

def preprocessFaces(dataPath):
    dataMatrix = []
    targetMatrix = []
    for fileName in os.listdir(dataPath):
        if fileName.lower() != "readme.txt":
            subjectId = int(fileName.split('.')[0][7:])
            imagePath = os.path.join(dataPath, fileName)
            image = Image.open(imagePath).convert("L") 
            resizedImage = image.resize((40, 40))
            flattenedImage = np.array(resizedImage).flatten()
            dataMatrix.append(flattenedImage)
            targetMatrix.append(subjectId)
    dataMatrix = np.array(dataMatrix)
    targetMatrix = np.array(targetMatrix)
    return dataMatrix, targetMatrix

def confusionMatrix(yTrue, yPred):
    confusionMatrix = np.zeros((15, 15), dtype=int)
    for i in range(len(yTrue)):
        trueClass = yTrue[i]
        predictedClass = yPred[i]
        confusionMatrix[trueClass - 1, predictedClass - 1] += 1
    return confusionMatrix

dataPath = './yalefaces'
xData, yData = preprocessFaces(dataPath)
shuffledIndices = np.random.permutation(len(xData))
trainIndices = []
validIndices = []
for personId in np.unique(yData):
    personIndices = np.where(yData == personId)[0]
    np.random.shuffle(personIndices)
    splitPoint = int(np.ceil(2/3 * len(personIndices)))
    trainIndices.extend(personIndices[:splitPoint])
    validIndices.extend(personIndices[splitPoint:])

xTrain = xData[trainIndices]
yTrain = yData[trainIndices]
xValid = xData[validIndices]
yValid = yData[validIndices]

knnPred = myKNN(xTrain, yTrain, xValid, k=3)
knnAccuracy = np.sum(knnPred == yValid) / len(yValid)
knnConfusionMatrix = confusionMatrix(yValid, knnPred)

print("KNN")
print("Validation Accuracy:", knnAccuracy)
print("Confusion Matrix:")
print(knnConfusionMatrix)

dt = createDecisionTree(xTrain, yTrain, maxDepth=5)
dtPred = [predictTree(dt, sample) for sample in xValid]
treeAccuracy = np.sum(dtPred == yValid) / len(yValid)
treeConfusionMatrix = confusionMatrix(yValid, dtPred)

print("Decision Tree")
print("Validation Accuracy:", treeAccuracy)
print("Confusion Matrix:")
print(treeConfusionMatrix)
