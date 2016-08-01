import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# A new column deck is created.
train["Deck"] = train["Cabin"].map(deck_num)

# The missing values (NaN values) of the columns are replaced with the median.
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Fare"] = train["Fare"].fillna(train["Fare"].median())

# The "Embarked" classes in the train table are converted to integers.
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

# The missing values are replaced with 0.
train["Embarked"] = train["Embarked"].fillna(0)

# A new column "family_size" is added to the tables.
train["family_size"] = train["SibSp"] + train["Parch"] + 1

features_list_1 = ["Sex", "family_size", "Age", "Survived"]
features_list_2 = ["Sex", "Pclass", "Deck", "Survived"]
features_list_3 = ["Sex", "Fare", "Embarked", "Survived"]

data1 = train[features_list_1]

sns.pairplot(data1, hue='Sex')

plt.show()

data2 = train[features_list_2]

sns.pairplot(data2, hue='Sex')

plt.show()

data3 = train[features_list_3]

sns.pairplot(data3, hue='Sex')

plt.show()
