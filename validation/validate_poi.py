#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""
import os
import joblib
import sys
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

data_dict = joblib.load(open("../final_project/final_project_dataset.pkl", "rb") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
y, X = targetFeatureSplit(data)


### Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.3, random_state=42)


### Build Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("accuracy: ", acc)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np


scores=[]
kf = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index,test_index in kf.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)
    
    X_train, X_test, y_train, y_test = np.array(X)[train_index], np.array(X)[test_index], np.array(y)[train_index], np.array(y)[test_index]
    
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

print("\n ###\nKFold Scores Mean: ", np.mean(scores))

print("\n ###\ncross_val_score: ", cross_val_score(clf, X, y, cv=10))
