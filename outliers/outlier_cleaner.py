#!/usr/bin/python

import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    # Create cleaned data list
    for p, age, worth in zip(predictions, ages, net_worths):
        cleaned_data.append( (age[0], worth[0], worth[0]-p[0]) )
            
            
    # Sort tuple list by Nth element of tuple
    cleaned_data.sort(key = lambda x: x[2], reverse=True)
    
    # index to remove by
    remove_index = len(cleaned_data) - round(len(cleaned_data)*.1)
    
    return cleaned_data[:remove_index]

