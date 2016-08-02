import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn import ensemble, grid_search
from sklearn.metrics import accuracy_score

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
train["Fare"] = train["Fare"].fillna(train["Fare"].median())
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# The "Embarked" classes in the train table are converted to integers.
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# The "Embarked" classes in the test table are converted to integers.
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# The missing values are replaced with 0.
train["Embarked"] = train["Embarked"].fillna(0)
test["Embarked"] = test["Embarked"].fillna(0)

# A new column "family_size" is added to the tables.
train["family_size"] = train["SibSp"] + train["Parch"] + 1
test["family_size"] = test["SibSp"] + test["Parch"] + 1

X_vars = ['PassengerId', 'Pclass', 'Sex', 'Age', 'family_size', 'Fare', 'Deck',
    'Embarked']
y_vars = 'Survived'
X_train, X_test, y_train, y_test = train_test_split(train[X_vars],
    train[y_vars], random_state = 1)

param_grid = {'max_depth': list(range(2, 20))}


model = ensemble.RandomForestClassifier(n_estimators=100,
    min_samples_split=200, random_state=1)
grid = grid_search.GridSearchCV(model, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train, y_train)

print('Scores:', grid.grid_scores_)
print('Best score:', grid.best_score_)
print('Best params:', grid.best_params_)
model = grid.best_estimator_

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print('accuracy score of prediction:', accuracy_score(y_test, y_pred))

# Creates a data frame with "PassangerId" and "Survived" (= prediction) columns.
PassengerId = np.array(X_test["PassengerId"]).astype(int)
solution = pd.DataFrame(y_pred, PassengerId, columns = ["Survived"])

# Solution is converted to CSV file.
solution.to_csv('titanic_grid_search_rf_cl.csv', index_label = ["PassengerId"])
