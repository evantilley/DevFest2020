import numpy as np
import pandas as pandas
import pymongo
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
from pymongo import MongoClient
from pprint import pprint
import pickle
import json
import csv
import time

# TODO: have node js add csv and json to
#  directory then execute script

def main():
    while(1):
        client = pymongo.MongoClient('mongodb://localhost:27017/')
        db = client["MachineLearning"]
        collection1 = db["Data"]
        collection3 = db["Test"]
        while collection1.count() < 1 or collection3 < 1:
            pass
        time.sleep(1)

        x = collection1.find_one()

        collection2 = db[x['dataset']]

        dataset = x['dataset']
        attributes = x['attributes']
        predict = x['predict']
        attributes.append(predict)

        cursor = collection2.find()
        mongo_docs = list(cursor)
        docs = pandas.DataFrame(columns = [])

        for num, doc in enumerate(mongo_docs):
            doc["_id"] = (doc["_id"])
            doc_id = doc["_id"]
            series_obj = pandas.Series(doc,name = doc_id)
            docs = docs.append(series_obj)

        docs.to_csv("susling.csv")

        f = open("ha.csv", "wt")
        with open("susling.csv", 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:
                for i in line:
                    f.write(str(i))
                    f.write('\n')

        count1 = 0
        count2 = 1
        test = False
        f = open("ha.csv", "r").readlines()
        with open("final.csv", "w") as outfile:
            for index, line in enumerate(f):
                # if (index != 1 and index != 3 and index != 4 and index != 6 and index != 7 and index != 9 and index != 10 and index != 12 and index != 13):
                #     outfile.write(line)
                if (index == 1):
                    pass
                if (index != count1 and index != count2):
                    outfile.write(line)
                    count1 += 3
                    count2 += 3



        f = open("final.csv")
        data = pandas.read_csv(f, sep=";")
        data = data[attributes]
        data = shuffle(data)

        x = np.array(data.drop([predict], 1))
        y = np.array(data[predict])
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

        best = 0
        for _ in range(20):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

            linear = linear_model.LinearRegression()

            linear.fit(x_train, y_train)
            acc = linear.score(x_test, y_test)
            print("Accuracy: " + str(acc))

            if acc > best:
                best = acc
                with open("model.pickle", "wb") as f:
                    pickle.dump(linear, f)

        pickle_in = open("model.pickle", "rb")
        linear = pickle.load(pickle_in)

        print(best)
        predicted = linear.predict(x_test)
        for x in range(len(predicted)):
            print(predicted[x], x_test[x], y_test[x])

        outputString = "Your model has been trained with {} accuracy.".format(best)
        output = {
            'output': outputString
        }

        collection1.drop()
        collection2.drop()
        result = db.reviews.insert_one(output)
        print(result)

        while collection3.count() < 1:
            pass
        time.sleep(1)

        cursor = collection3.find()
        mongo_docs = list(cursor)
        docs = pandas.DataFrame(columns=[])

        for num, doc in enumerate(mongo_docs):
            doc["_id"] = (doc["_id"])
            doc_id = doc["_id"]
            series_obj = pandas.Series(doc, name=doc_id)
            docs = docs.append(series_obj)

        docs.to_csv("a.csv")

        f = open("b.csv", "wt")
        with open("a.csv", 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:
                for i in line:
                    f.write(str(i))
                    f.write('\n')

        count1 = 0
        count2 = 1
        test = False
        f = open("b.csv", "r").readlines()
        with open("c.csv", "w") as outfile:
            for index, line in enumerate(f):
                # if (index != 1 and index != 3 and index != 4 and index != 6 and index != 7 and index != 9 and index != 10 and index != 12 and index != 13):
                #     outfile.write(line)
                if (index == 1):
                    pass
                if (index != count1 and index != count2):
                    outfile.write(line)
                    count1 += 3
                    count2 += 3

        f = open("c.csv")
        data = pandas.read_csv(f, sep=";")
        data = data[attributes]
        x = np.array(data.drop([predict], 1))
        y = np.array(data[predict])
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=1.0)

        predicted = linear.predict(x_test)
        for x in range(len(predicted)):
            db.reviews.insert_one(predicted[x], x_test[x], y_test[x])

        collection3.drop()




if __name__== "__main__":
    main()
# TODO: send results back through node js sus stuff so it can be displayed on website