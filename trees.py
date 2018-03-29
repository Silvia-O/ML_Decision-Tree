from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]    # The last element of each feature vector is the label
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0    # Add the new label if it does not exist
        labelCounts[currentLabel] += 1
        shannonEnt = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / numEntries
            shannonEnt -= prob * log(prob, 2)
        return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])  # Add multiple values in another sequence at the end of the list
            retDataSet.append(reducedFeatVec)        # Add another sequence at the end of the list
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0] - 1)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for axis in range(numFeatures):
        featList = [featVec[axis] for featVec in dataSet]    # Get all vals in a list for this feature axis
        uniqueVals = set(featList)   # A set requires its elements different from one aother
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, axis, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = axis
    return bestFeature

def majorityCnt(classList):
    classCounts = {}
    for classLabel in classList:
        if classLabel not in classCounts.keys():
            classCounts[classLabel] = 0
        classCounts[classLabel] += 1
    sortedClassCounts = sorted(classCounts.iteritems(), key=operator.itemgetter(1), reverse=True)  # Keys are different from one another
    return sortedClassCounts[0][0]  # Return the classType which exits the most

def createTree(dataSet, featLabels):
    classList = [featVec[-1] for featVec in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = featLabels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(featLabels[bestFeat])
    featValues = [featVec[bestFeat] for featVec in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subFeatLabels = featLabels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subFeatLabels)
    return myTree

def classify(inputTree, featLabels, featVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if featVec[featIndex]  == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, featVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)