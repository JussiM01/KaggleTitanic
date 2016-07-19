import pandas as pd
import numpy as np
from sklearn import tree

# This line had to be added inorder for chained assignment to work properly.
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

# Tables for survival values and values of the features are created.
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# The missing values (NaN values) of the columns are replaced with the median.
trai["Fare"] = train["Fare"].fillna(train["Fare"].median())
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# The first version of the decision tree is created and fitted.
# (Other versions of the decision tree are in different files.)
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Creastes a table of test features.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Creates a prediction from the test features.
my_prediction = my_tree_one.predict(test_features)

# Creates a data frame with "PassangerId" and "Suvived" (= prediction) columns.
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Solution is converted to CSV file.
my_solution.to_csv('my_first_titanic_tree.csv', index_label = ["PassengerId"])
