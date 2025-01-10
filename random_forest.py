# playground for random forest
from decision_tree import RandomForest, DecisionTree
import pandas as pd
import numpy as np
from collections import deque


selected_features = 'Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked'
df = pd.read_csv('titanic/train.csv')[selected_features.split(',')].sample(200)

test_data = pd.read_csv('titanic/train.csv').iloc[:5]

trial_forest = RandomForest(train_data=df, decision_feature='Embarked', tree_count=1)
trial_forest.learn(thresholder='iter')

# print(trial_forest.forest_vote(test_data))

# tree = DecisionTree(train_data=df, decision_feature='Embarked', k=6)

# tree.learn(thresholder='iter')

# print(tree.decide(test_data))









