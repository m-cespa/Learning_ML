from decision_tree import DecisionTree
import pandas as pd

selected_features = 'Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked'
data = pd.read_csv('Random_Forest_from_scratch/titanic/train.csv')[selected_features.split(',')]

tree = DecisionTree(train_data=data, decision_feature='Survived', k=6, numerical_threshold_lim=1)
tree.learn(thresholder='iter_bf', min_child_nodes=10)
tree.print_tree()
