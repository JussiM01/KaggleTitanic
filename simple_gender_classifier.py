import pandas as pd

# This line had to be added inorder for chained assignment to work properly.
pd.options.mode.chained_assignment = None

# Not really used in this simple gender classifier, except for the realization
# that females are more likely to survive than males.
train = pd.read_csv('train.csv')

# The file test.csv is read with pandas.
test = pd.read_csv('test.csv')

# Creates and initializes the column "Survived".
test["Survived"] = 0

# Sets the value of "Survived" to 1 if female.
test["Survived"][test["Sex"] == 'female'] = 1

# Creates a table with columns "PassengerId" and "Survived" from test table.
solution = test[["PassengerId", "Survived"]]

# Converts the solution table into CSV file simple_gender_classifier.csv.
solution.to_csv('simple_gender_classifier.csv', index = False)
