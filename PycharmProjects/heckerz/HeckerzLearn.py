import numpy as np
import pandas as pd
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import json


# style.use("ggplot")

# TODO: have node js add csv and json to
#  directory then execute script

with open('input.json') as f:
    input = json.load(f)

dataset = input['dataset']
attributes = input['attributes']
predict = input['predict']

# dataset = "student-mat.csv"
# attributes = ["G1", "G2", "absences", "failures", "studytime", "G3"]
# predict = "G2"

data = pd.read_csv(dataset, sep=";")

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
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

# LOAD MODEL
pickle_in = open("studentgrades.pickle", "rb")
linear = pickle.load(pickle_in)


# print("-------------------------")
# print('Coefficient: \n', linear.coef_)
# print('Intercept: \n', linear.intercept_)
# print("-------------------------")

print(best)
predicted= linear.predict(x_test)
for x in range(len(predicted)):
    print(predicted[x], x_test[x], y_test[x])

# TODO: send results back through node js sus stuff so it can be displayed on website