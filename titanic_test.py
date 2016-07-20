import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# This line had to be added in order for chained assignment to work properly in
# Python 3 (in Python 2 the code probably works even without it).
pd.options.mode.chained_assignment = None

# The file train.csv is read with pandas.
train = pd.read_csv('train.csv')

# The file test.csv is read with pandas.
test = pd.read_csv('test.csv')

def print_columns(column_string):
    print('train ' + column_string + 'values')
    print(train[column_string])
    print('')
    print('test ' + column_string + 'values')
    print(test[column_string])
    print('')
