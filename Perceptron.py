import json
import re
import csv
import numpy.matlib
import numpy as np
import random

keywords = None
maxIter = 20

def getLabels(data, positive, negative):
    subset = [0] * len(data)
    for s, d in enumerate(data):
        if d[0] == positive:
            subset[s] = 1
        elif d[0] == negative:
            subset[s] = -1
        else:
            print("Error, unreachable state")
            exit(-1)
    return subset

def getFullLabels(data):
    labels = [0] * len(data)
    for i, d in enumerate(data):
        labels[i] = d[0]
    return labels

def getFeatures(data):
    # subset = data
    # for i in range(len(subset)):
    #     del subset[i][0]
    # return subset
    subset = [[] for i in range(len(data))]
    # subset = [[]] * len(data)
    for i in range(len(data)):
        subset[i] = data[i][1:]
    return subset

def subsetData(data, positive, negative):
    subset = data
    # for i in range(len(subset)):
    #     if not (subset[i][0] == positive or subset[i][0] == negative):
    #         del subset[i]
    #         i = i - 1
    # subset = [x for x in subset if (x[0] == positive or x[0] == negative)]
    return list(filter(lambda x: x[0] == positive or x[0] == negative, subset))

def extractTestData(entry):
    wordData = [0] * (len(keywords) + 4)
    text = entry["text"]
    wordData[0] = text.count("!")  # number of exclamation points
    wordData[1] = text.count("$")  # number of $ signs
    text = re.findall(r"\w+", entry["text"])
    wordData[2] = len(text)  # how many words in review
    for w in text:
        if w.isupper() and len(w) != 1 and (not any(char.isdigit() for char in w)):  # if word is all caps, not a single letter, and not numeric. note: might want to change not numeric detection
            wordData[3] += 1  # increase all caps feature
        w = w.lower()
        for i, k in enumerate(keywords):
            if w == k[0]:
                wordData[i + 4] += 1
                break
    return wordData

# Returns a vector of occurrences of words in keywords and some other special features
# Ex: keywords = ["good", "bad", "okay"]
#     entry = "the food was good for a good price of $10, but the service was only okay. DEFINITELY worth it I would say"
#     wordData = [number of !, number of $, length in words, number of all caps, number of "good", number of "bad", number of "okay"]
#     wordData = [0, 1, 22, 1, 2, 0, 1]
def extractTrainingData(entry):
    wordData = [0] * (len(keywords) + 5)
    text = entry["text"]
    wordData[0] = entry["stars"]
    wordData[1] = text.count("!") # number of exclamation points
    wordData[2] = text.count("$") # number of $ signs
    text = re.findall(r"\w+", entry["text"])
    wordData[3] = len(text) # how many words in review
    for w in text:
        if w.isupper() and len(w) != 1 and (not any(char.isdigit() for char in w)): # if word is all caps, not a single letter, and not numeric. note: might want to change not numeric detection
            wordData[4] += 1 # increase all caps feature
        w = w.lower()
        for i, k in enumerate(keywords):
            if w == k[0]:
                wordData[i + 5] += 1
                break
    return wordData

def subsetTrain(trainingSet, labels, iter):
    w = [0] * (len(trainingSet[0]))
    w = np.asarray(w, dtype=np.float32)
    b = 0
    for i in range(iter):
        combo = list(zip(trainingSet, labels))
        random.shuffle(combo)
        shuffledTrain, shuffledLabels = zip(*combo)
        for x, y in zip(shuffledTrain, shuffledLabels):
            xa = np.asarray(x, dtype=np.float32)
            a = np.dot(w, xa) + b
            if (y * a) <= 0:
                w = np.add(w, y*xa)
                b = b + y
    w = np.insert(w, 0, b, axis=0)
    return w

def train(fullData):
    classifiers = [[[] for i in range(5)] for j in range(5)]
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            else:
                subset = subsetData(fullData, i + 1, j + 1)
                subsetLabels = getLabels(subset, i + 1, j + 1)
                subsetFeatures = getFeatures(subset)
                classifiers[i][j] = subsetTrain(subsetFeatures, subsetLabels, maxIter)
            print("Trained classifier " + str(i) + "/" + str(j))
    return classifiers

# predict an instance with classifier w
def predict(w, instance):
    a = np.dot(w[1:], instance) + w[0]
    if a >= 0:
        return 1
    else:
        return -1

# perform ava
def fullPredict(classifiers, instances):
    predictions = [0] * len(instances)
    for p, i in enumerate(instances):
        individualPredictions = [0] * 5
        for j in range(5):
            for k in range(5):
                if j == k:
                    continue
                prediction = predict(classifiers[j][k], i)
                individualPredictions[j] = individualPredictions[j] + prediction
                individualPredictions[k] = individualPredictions[k] - prediction
        predictions[p] = individualPredictions.index(max(individualPredictions)) + 1
    return predictions

def main():
    # load keyword features
    global keywords
    with open('../keywords.csv', 'r') as kwords: # keywords.csv is a csv file that lists each word feature to extract
        reader = csv.reader(kwords)
        keywords = list(reader)

    # extract features from training set
    fullData = []
    with open("../data_train.json") as jsonFile:
        data = json.load(jsonFile)

        fullData = [[] for i in range(len(data))] # is a list of all instance feature vectors
        for t, d in enumerate(data):
            fullData[t] = extractTrainingData(d)

    # train algorithm and predict training set
    print("Feature extraction finished")
    classifiers = train(fullData)
    trainingPredictions = fullPredict(classifiers, getFeatures(fullData))
    print("Predictions finished")
    trueLabels = getFullLabels(fullData)

    # determine training set accuracy
    accuracy = 0
    for p, t in zip(trainingPredictions, trueLabels):
        if p == t:
            accuracy = accuracy + 1
    print(accuracy / len(fullData))

    # write training predictions
    with open("trainingPredictions.csv", 'w', newline='') as output:
        wr = csv.writer(output, quoting=csv.QUOTE_ALL)
        for p in trainingPredictions:
            wr.writerow([p])

    # extract test instance features
    testInstances = []
    with open("../data_test_wo_label.json") as jsonFile:
        testData = json.load(jsonFile)

        testInstances = [[] for i in range(len(testData))] # is a list of all instance feature vectors
        for t, d in enumerate(testData):
            testInstances[t] = extractTestData(d)

    # predict and write test predictions
    testPredictions = fullPredict(classifiers, testInstances)
    with open("testPredictions.csv", 'w', newline='') as output:
        wr = csv.writer(output, quoting=csv.QUOTE_ALL)
        for p in testPredictions:
            wr.writerow([p])
main()