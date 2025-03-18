from decision_tree import DecisionTree
import pandas as pd
import numpy as np

# selected_features = 'Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked'
# data = pd.read_csv('Random_Forest_from_scratch/titanic/train.csv')[selected_features.split(',')]

# tree = DecisionTree(train_data=data, decision_feature='Embarked', k=7, numerical_threshold_lim=1)
# tree.learn(thresholder='iter_bf', min_child_nodes=10)
# tree.print_tree()


data = pd.read_csv('Random_Forest_from_scratch/denmark_waste/2013_data_totalwaste.csv').drop(['Location'], axis=1)

train_data = data.iloc[4:].copy()


tree = DecisionTree(train_data=train_data, decision_feature='TOTAL HOUSEHOLD WASTE', k=17, numerical_threshold_lim=1)

tree.learn(thresholder='iter_bf', min_child_nodes=5)

tree.print_tree()

# testing_rows = data.iloc[2:4]

# print(tree.decide(testing_rows))
