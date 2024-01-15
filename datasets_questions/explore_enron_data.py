#!/usr/bin/python3

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import joblib
import pandas as pd
import numpy as np

enron_data = joblib.load(open("../final_project/final_project_dataset.pkl", "rb"))

enron_df = pd.DataFrame.from_dict(enron_data, orient='index')
enron_df.reset_index(inplace=True)
enron_df.rename(columns={'index':'name'}, inplace=True)

# Size of Enron Dataset
data_rows = enron_df.shape[0]
print('Number of Rows: {}'.format(data_rows))

# Number of Features
print('Number of Features: {}'.format(enron_df.shape[1]))

# Features
print(enron_df.columns)

# How many POIs in the data
poi_df = enron_df.query('poi == 1')
print('POIs in Data: {}'.format(len(poi_df)))

# How many POIs total
file_name = "../final_project/poi_names.txt"

with open(file_name, 'r') as fp:
    names = len(fp.readlines())
    print('Total POI: {}'.format(names-2))
    
# In Csuite, who made the most
csuite = ['FASTOW ANDREW S', 'SKILLING JEFFREY K', 'LAY KENNETH L']
print(enron_df[enron_df['name'].isin(csuite)][['name', 'total_payments']])

# Fix NaN values stored as strings
enron_df.replace({'NaN':np.nan}, inplace=True)

# Missing Salary
print("# of missing salary: {}".format(enron_df.salary.isna().sum()))

# Missing email
print("# of missing emails: {}".format(enron_df.email_address.isna().sum()))

# Proportion missing "Total Payments"
print("% missing Total Payments: {}".format(enron_df.total_payments.isna().sum()/data_rows))

# Proportion missing "Total Payments" for POIs
print("% POI missing Total Payments: {}".format(poi_df.total_payments.isna().sum()/len(poi_df)))
