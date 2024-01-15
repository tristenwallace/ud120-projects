#!/usr/bin/python3
import os
import joblib
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("../tools/"))
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = joblib.load( open("../final_project/final_project_dataset.pkl", "rb") )
features = ["salary", "bonus"]

data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)



print(data)
### your code below
plt.scatter(data[:, 0], data[:, 1])
plt.show()