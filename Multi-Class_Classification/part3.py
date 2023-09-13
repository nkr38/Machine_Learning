# Student name(s): Noah Robinson
# Date modified: 8/15/23

import numpy as np
np.random.seed(0)

class TreeNode:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.left = None
        self.right = None

def findEntropy(labels):
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropyValue = -np.sum(probabilities * np.log2(probabilities))
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
        # bincount was giving casting error so using unique here
        uniqueLabels, counts = np.unique(labels, return_counts=True)
        return TreeNode(None, uniqueLabels[np.argmax(counts)])
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

data = np.genfromtxt('CTG.csv', delimiter=',', skip_header=2, usecols=range(1, 23))
x = data[:, :-2]
y = data[:, -1]

shuffledIndices = np.random.permutation(len(x))
xShuffled = x[shuffledIndices]
yShuffled = y[shuffledIndices]

splitPoint = int(np.ceil(2/3 * len(x)))
xTrain = xShuffled[:splitPoint]
yTrain = yShuffled[:splitPoint]
xValid = xShuffled[splitPoint:]
yValid = yShuffled[splitPoint:]

decisionTree = createDecisionTree(xTrain, yTrain, maxDepth=5)
validationPredictions = [predictTree(decisionTree, sample) for sample in xValid]
validationAccuracy = np.sum(validationPredictions == yValid) / len(yValid)

print("Validation Accuracy: ",validationAccuracy)
confusionMatrix = np.zeros((3, 3), dtype=int)

for i in range(len(yValid)):
    trueClass = yValid[i]
    predictedClass = validationPredictions[i]
    trueIndex = int(trueClass) - 1
    predictedIndex = int(predictedClass) - 1
    confusionMatrix[trueIndex, predictedIndex] += 1

print("\nConfusion Matrix:")
print(confusionMatrix)