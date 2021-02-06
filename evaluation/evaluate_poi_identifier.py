#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
import numpy as np

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.30, random_state=42)

regressor = DecisionTreeRegressor()
regressor.fit(features_train, labels_train)
print "Accuracy = ", regressor.score(features_test, labels_test)
print "POIs: ", regressor.predict(features_test)
print "Num of people in test set: ", len(features_test)
print "True Positive? : ", np.multiply(regressor.predict(features_test), labels_test)
print "Precision: ", precision_score(labels_test, regressor.predict(features_test))
print "Recall: ", recall_score(labels_test, regressor.predict(features_test))
