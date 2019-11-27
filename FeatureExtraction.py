import json
import re
import csv

keywords = None

# Returns a vector of occurrences of words in keywords
# Ex: keywords = ["good", "bad", "okay"]
#     entry = "the food was good and the prices were good too, but the service was only okay"
#     wordData = [2, 0, 1]
def getVector(entry):
    wordData = [0] * len(keywords)
    text = entry["text"]
    text = re.findall(r"\w+", entry["text"])
    for w in text:
        w = w.lower()
        for i, k in enumerate(keywords):
            if w == k[0]:
                wordData[i] += 1
                break
    return wordData

def main():
    global keywords
    with open('../keywords.csv', 'r') as kwords: # keywords.csv is a csv file that lists each word feature to extract
        reader = csv.reader(kwords)
        keywords = list(reader)

    # Get word count vector
    with open("../data_train.json") as jsonFile:
        data = json.load(jsonFile)
        vector = getVector(data[0])
        print(vector)

        # Loop version
        # for d in data:
        #     vector = getVector(d)
        #     # print(vector) # You probably don't want to print this

main()