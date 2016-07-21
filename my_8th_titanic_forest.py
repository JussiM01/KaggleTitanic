import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# This line had to be added in order for chained assignment to work properly in
# Python 3 (in Python 2 the code probably works even without it).
pd.options.mode.chained_assignment = None

decks = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
    'T': 8
}

def deck_num(cabin):
    if isinstance(cabin, str): return decks[cabin[0]]
    return 0

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

# A new column deck is created.
train["Deck"] = train["Cabin"].map(deck_num)
test["Deck"] = test["Cabin"].map(deck_num)

# The missing values (NaN values) of the columns are replaced with the median.
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# A new column "family_size" is added to the tables.
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1

features_list = ["Sex", "family_size", "Pclass"]

# Tables for survival values and values of the features are created.
target = train["Survived"].values
features_forest = train[features_list].values

# Function which is used in spliting the train data in to training set and cross
# validation set.
def split_data(data, ratio):
    import random
    random.seed(1)
    size = int(data.shape[0] * ratio)
    rows = random.sample(list(data.index), size)
    a = data.ix[rows]
    b = data.drop(rows)
    return a, b

def train_classifier(hp, training_features, training_target):
    forest = RandomForestClassifier(
        max_depth = hp[0], min_samples_split = hp[1],
        n_estimators = 100, random_state = 1)
    return forest.fit(training_features, training_target)

def hyper_param_optim(data, features_list):
    # Training set and cross validation set.
    data_0, data_1 = split_data(train, 0.5)

    best_parameters = (0, 0)
    best_result = 0.0
    hp_target = data_0["Survived"].values
    hp_features = data_0[features_list].values

    # Parameters for hyper-parameter optimization.
    hyp_params = {(i, j) for i in range(2, 20) for j in range(2, 10)}

    for hp in hyp_params:
        forest = train_classifier(hp, hp_features, hp_target)
        result = cross_validate(forest, features_list, data_1)
        if result > best_result:
            best_result = result
            best_parameters = hp
    print('best parameters: ', best_parameters)
    return best_parameters

def cross_validate(forest, features_list, validation_data):
    data = validation_data.copy()
    features = data[features_list].values
    prediction = forest.predict(features)
    data['Prediction'] = prediction
    p = (data['Prediction'] == data['Survived']).value_counts(normalize = True)
    return p[True]

params = hyper_param_optim(train, features_list)

# Building and fitting the forest.
forest = RandomForestClassifier(max_depth = params[0],
        min_samples_split = params[1],
        n_estimators = 100, random_state = 1)
my_forest = forest.fit(features_forest, target)

# Creates a table of test features.
test_features = test[features_list].values

# Creates a prediction from the test features.
pred_forest = my_forest.predict(test_features)

# Creates a data frame with "PassangerId" and "Survived" (= prediction) columns.
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(pred_forest, PassengerId, columns = ["Survived"])

# Solution is converted to CSV file.
my_solution.to_csv('8th_titanic_forest.csv', index_label = ["PassengerId"])
