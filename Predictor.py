# Jacob Baginski Doar
# 11/22/2019
# Predictor.py

# Implimentation of a neural network used to predict ratings on yelp reviews

# Run in the command line with the format
# python DataExtraction.py File -Mode
# Where mode can be train or test

import math
import json
import re
import csv

# Returns a vector of occurrences of words in keywords and some other special features
# Ex: keywords = ["good", "bad", "okay"]
#     entry = "the food was good for a good price of $10, but the service was only okay. DEFINITELY worth it I would say"
#     wordData = [number of !, number of $, length in words, number of all caps, number of "good", number of "bad", number of "okay"]
#     wordData = [0, 1, 22, 1, 2, 0, 1]
def extractFeatures(entry):
    wordData = [0] * (len(keywords) + 4)
    text = entry["text"]
    wordData[0] = text.count("!") # number of exclamation points
    wordData[1] = text.count("$") # number of $ signs
    text = re.findall(r"\w+", entry["text"])
    wordData[2] = len(text) # how many words in review
    for w in text:
        if w.isupper() and len(w) != 1 and (not any(char.isdigit() for char in w)): # if word is all caps, not a single letter, and not numeric. note: might want to change not numeric detection
            wordData[3] += 1 # increase all caps feature
        w = w.lower()
        for i, k in enumerate(keywords):
            if w == k[0]:
                wordData[i + 4] += 1
                break
    return wordData

class LogisticRegression:
    # Constructor
    def __init__(self, n):
        self.weights = [0] * n
        self.rate = 200
        self.iterations = 200

    def sigmoid(self, z):
        return 1 / (1 + math.exp(-z))

    def activation(self, x):
        dot = 0.0
        for i in range(len(x)):
            dot += self.weights[i] * x[i]
        return dot

    def probPred1(self, x):
        return self.sigmoid(self.activation(x))

    def predict(self, x):
        if self.probPred1(x) >= 0.5:
            return 1
        else:
            return 0

    # Takes in an array of LRInstances
    def printPerformance(self, instances):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        
        for i in range(len(instances)):
            label = instances[i].label
            x = instances[i].x
            prediction = self.predict(x)

            if label == 1 and prediction == 1:
                TP += 1
            elif label == 1 and prediction == 0:
                FN += 1
            elif label == 0 and prediction == 1:
                FP += 1
            elif label == 0 and prediction == 0:
                TN += 1

        acc = (TP + TN) / len(instances)

        #p_pos = TP / (TP + FP)
        #r_pos = TP / (TP + FN)
        #f_pos = (2 * p_pos * r_pos) / (p_pos + r_pos)

        #p_neg = TN / (TN + FN)
        #r_neg = TN / (TN + FP)
        #f_neg = (2 * p_neg * r_neg) / (p_neg + r_neg)

        print("Accuracy" + str(acc))
        #print("P, R, and F1 score of the positive class=" + str(p_pos) + " " + str(r_pos) + " " + str(f_pos))
        #print("P, R, and F1 score of the negative class=" + str(p_neg) + " " + str(r_neg) + " " + str(f_neg))
        print("Confusion Matrix")
        print(str(TP) + "   " + str(FN))
        print(str(FP) + "   " + str(FN))

    # Takes in an array of LRInstance
    def train(self, instances):
        for i in range(self.iterations):
            for j in range(len(instances)):
                label = instances[j].label
                x = instances[j].x
                gradient = label - self.probPred1(x)

                for k in range(len(self.weights)):
                    self.weights[k] += self.rate * gradient * x[k]
                    
class LRInstance:
    def __init__(self, label, x):
        self.label = label
        self.x = x

# Return an array of LRInstances 
def readDataSet(path, start, end):
    instances = []

    with open(path) as jsonFile:
        data = json.load(jsonFile)
        
        for i in range(start, end):
            label = data[i]["stars"]
            x = extractFeatures(data[i])

            instance = LRInstance(label, x)
            instances.append(instance)

    return instances
            
        

def main():
    global keywords
    with open('./keywords.csv', 'r') as kwords: # keywords.csv is a csv file that lists each word feature to extract
        reader = csv.reader(kwords)
        keywords = list(reader)
    
    # Read in data
    # Both data instances are arrays of LRInstances
    trainInstances = readDataSet("../data_train.json", 0, 1000)
    testInstances = readDataSet("../data_train.json", 1000, 1100)

    #print(len(trainInstances))
    #print(len(testInstances))

    n = len(trainInstances[0].x)
    LR = LogisticRegression(n)
    
    # Train
    LR.train(trainInstances)

    # Print Accuracy
    print("Training Data Performance")
    LR.printPerformance(trainInstances)

    print("Test Data Performance")
    LR.printPerformance(testInstances)
    
main()

