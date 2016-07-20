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

# The "male" and "female" values in the train table are converted to integers.
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# The "male" and "female" values in the train table are converted to integers.
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# The missing values (NaN values) of the columns are replaced with the median.
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# Tables for survival values and values of the features are created.
target = train["Survived"].values
features_forest = train[
        ["Age", "Sex"]].values

# Building and fitting the forest.
forest = RandomForestClassifier(max_depth = 10, min_samples_split = 2,
        n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Creastes a table of test features.
test_features = test[
        ["Age", "Sex"]].values

# Creates a prediction from the test features.
pred_forest = my_forest.predict(test_features)

# Creates a data frame with "PassangerId" and "Suvived" (= prediction) columns.
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

# Solution is converted to CSV file.
my_solution.to_csv('my_2nd_titanic_forest.csv', index_label = ["PassengerId"])
