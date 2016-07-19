import pandas as pd
import numpy as np
import cvs as cvs
from sklearn.ensamble import RandomForestClassifier

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

# Tables for survival values and values of the features are created.
target = train["Survived"].values
features_forest = train[
        ["Pclass", "Age", "Sex" "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting the forest.
forest = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 2,
        n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Creastes a table of test features.
test_features = test[
        ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Creates a prediction from the test features.
pred_forest = my_forest.predict(test_features)

# Creates a data frame with "PassangerId" and "Suvived" (= prediction) columns.
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

# Solution is converted to CSV file.
my_solution.to_csv('my_titanic_forest.csv', index_label = ["PassengerId"])
