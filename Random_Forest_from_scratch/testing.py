from decision_tree import DecisionTree
import pandas as pd
import numpy as np

# selected_features = 'Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked'
# data = pd.read_csv('Random_Forest_from_scratch/titanic/train.csv')[selected_features.split(',')]

# tree = DecisionTree(train_data=data, decision_feature='Embarked', k=7, numerical_threshold_lim=1)
# tree.learn(thresholder='iter_bf', min_child_nodes=10)
# tree.print_tree()


data = pd.read_csv('Random_Forest_from_scratch/2013_data_totalwaste.csv').drop(['Location'], axis=1)


tree = DecisionTree(train_data=data, decision_feature='TOTAL HOUSEHOLD WASTE', k=10, numerical_threshold_lim=2)

tree.learn(thresholder='iter_bf', min_child_nodes=5)

tree.print_tree()

testing_rows = data.iloc[4:6]

print(tree.decide(testing_rows))
