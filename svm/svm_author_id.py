#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
from sklearn import svm

clf = svm.SVC(kernel='rbf', C= 10000)

#print("Fitting the model...")
#t0 = time()
#clf.fit(features_train, labels_train)
#print("Training Time:", round(time()-t0, 3), "s")


#print(clf.score(features_test, labels_test))


#########################################################

#########################################################
'''
Use the below training set for faster fit, but lower accuracy (by ~10%).
'''

#features_train = features_train[:int(len(features_train)/100)]
#labels_train = labels_train[:int(len(labels_train)/100)]

print("Fitting the model...")
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

print("Predicting...")
t0 = time()
predictions = clf.predict(features_test)        
print("Prediction Time:", round(time()-t0, 3), "s\n")

print("{} of the test emails belong to Chris".format(predictions.sum()))

print("\naccuracy is: {}".format(clf.score(features_test, labels_test)))
#########################################################
