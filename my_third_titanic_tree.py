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

# The "Embarked" classes in the train table are converted to integers.
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# The "Embarked" classes in the test table are converted to integers.
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# The missing values (NaN values) of the columns are replaced with the median.
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())
train["Embarked"] = train["Embarked"].fillna(0)
test["Embarked"] = test["Embarked"].fillna(0)

# Creates new tables with the added feature "family_size".
train_two = train.copy()
test_two = test.copy()
train_two["family_size"] = train_two["SibSp"] + train_two["Parch"] + 1
test_two["family_size"] = test_two["SibSp"] + test_two["Parch"] + 1

# Creates new feature set with added feature "family_size".
features_three = train_two[
    ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "family_size"]].values

# Table for survival values is created.
target = train_two["Survived"].values

# The first version of the decision tree is created and fitted.
# (Other versions of the decision tree are in different files.)
my_tree_three = tree.DecisionTreeClassifier()
my_tree_two = my_tree_two.fit(features_three, target)

# Creates a prediction from the test features.
my_prediction = my_tree_three.predict(features_three)

# Creates a data frame with "PassangerId" and "Suvived" (= prediction) columns.
PassengerId = np.array(test_two["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Solution is converted to CSV file.
my_solution.to_csv('my_third_titanic_tree.csv', index_label = ["PassengerId"])
