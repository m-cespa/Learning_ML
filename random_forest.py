# playground for random forest
from decision_tree import DecisionTree
import pandas as pd
import numpy as np
from collections import deque

def print_tree_bfs(root):
    """
    Prints the decision tree in a breadth-first manner, following the tree shape with indentation.
    
    Args:
        root: The root TreeNode of the tree.
    """
    if root is None:
        return

    # Use a queue to perform BFS
    queue = deque([(root, 0)])  # Store (node, depth)
    current_depth = 0
    current_level_nodes = []

    while queue:
        node, depth = queue.popleft()

        # If we've reached a new depth level, print the previous level's nodes and reset
        if depth > current_depth:
            # Print current level nodes with spacing to align them correctly
            print("   ".join(current_level_nodes))
            current_level_nodes = []
            current_depth = depth
        
        # Add the current node to the level's output list
        current_level_nodes.append(str(node.label))

        # Enqueue the children of the current node
        if node.children:
            for child in node.children:
                queue.append((child, depth + 1))
    
    # Print the last level
    if current_level_nodes:
        print("   ".join(current_level_nodes))



selected_features = 'Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked'
df = pd.read_csv('titanic/train.csv')[selected_features.split(',')].sample(200)

tree = DecisionTree(train_data=df, k=6)

tree.learn(thresholder='iter')

print_tree_bfs(tree.root)

test_data = pd.read_csv('titanic/train.csv').iloc[:1]

if tree.decide(test_data=test_data):
    print('\n Survivor!!')
else:
    print('\n Died :/')









