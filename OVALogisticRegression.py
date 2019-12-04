# Jacob Baginski Doar
# 11/22/2019
# OVALogisticRegression.py

# Implimentation of OVA logistic regression used to predict ratings on yelp reviews

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

class OVALogisticRegression:
    # Constructor
    def __init__(self, n):
        # Create a 5 x n array of weights
        self.weights = [0] * 5
        for i in range(5):
            self.weights[i] = [0] * n
        self.rate = 200
        self.iterations = 200

    def sigmoid(self, x):
        try:
            return 1 / (1 + math.exp(-x))
        except:
            return 0

    def activation(self, x, k):
        dot = 0.0
        for i in range(len(x)):
            dot += self.weights[k][i] * x[i]
        return dot

    # Return a probability prediction of instance x being in class k
    def probPred1(self, x, k):
        return self.sigmoid(self.activation(x, k))

    # Return the predicted class for instance x 
    def predict(self, x):
        scores = [0] * 5
        for i in range(5):
            scores[i] = self.probPred1(x, i)
        
        return scores.index(max(index)) + 1

    # Takes in an array of Instances
    def printPerformance(self, instances):
        TP = [0] * 5
        TN = [0] * 5
        FP = [0] * 5
        FN = [0] * 5
        
        for k in range(5):
            for i in range(len(instances)):
                label = 1 if instances[i].label == k + 1 else 0
                x = instances[i].x
                prediction = self.predict(x)

                if label == 1 and k + 1 == prediction:
                    TP[k] += 1
                elif label == 1 and k + 1 != prediction:
                    FN[k] += 1
                elif label == 0 and k + 1 == prediction:
                    TN[k] += 1
                elif label == 0 and k + 1 != prediction:
                    FN[k] += 1

        acc = [0] * 5
        for i in range(5):
            acc[i] = (TP[i] + TN[i]) / len(instances)

        for k in range(5):  
            print("Classifier for class: " + str(k + 1.0))
            print("Accuracy" + str(acc[k]))
            print("Confusion Matrix")
            print(str(TP[k]) + "   " + str(FN[k]))
            print(str(FP[k]) + "   " + str(TN[k]))
            print("")
        

    # Takes in an array of Instance
    def train(self, instances):
        # Train 5 classifiers
        for k in range(5):
            print("Training classifier: " + str(k + 1))
            for i in range(self.iterations):
                for j in range(len(instances)):
                    label = 1 if instances[j].label == k + 1 else 0
                    x = instances[j].x
                    gradient = label - self.probPred1(x, k)

                    for w in range(len(self.weights[k])):
                        self.weights[k][w] += self.rate * gradient * x[w]

    # Outputs test results to a file
    def test(self, instances):
        print("Hello world")

class Instance:
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

            instance = Instance(label, x)
            instances.append(instance)

    return instances
           
def main():
    global keywords
    with open('./keywords.csv', 'r') as kwords: # keywords.csv is a csv file that lists each word feature to extract
        reader = csv.reader(kwords)
        keywords = list(reader)
    
    # Read in data
    # Both data instances are arrays of Instances
    print("Extracting training instances")
    trainInstances = readDataSet("data_train.json", 0, 100)
    print("Extracting testing instances")
    testInstances = readDataSet("data_train.json", 1000, 1100)
    
    n = len(trainInstances[0].x)
    OVALR = OVALogisticRegression(n)
    
    # Train
    print("Training")
    OVALR.train(trainInstances)

    # Print Accuracy
    print("Training Data Performance")
    OVALR.printPerformance(trainInstances)

    print("Test Data Performance")
    OVALR.printPerformance(testInstances)
    
main()

