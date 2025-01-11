# decision_tree.py
# pure python decision tree algorithm using ID3 partitioning
import numpy as np
from typing import List
import pandas as pd
import random
from collections import deque, Counter
import pickle
import os

class TreeNode:
    """
    Base class for decision tree nodes.
    """
    def __init__(self, label: str, children: List['TreeNode']=None):
        self.label = label
        self.children = children if children is not None else []

class FeatureNode(TreeNode):
    """
    Node representing the feature about which the split is made.
    """
    def __init__(self, label: str, feature_type: str, children: List[TreeNode]=None):
        super().__init__(label=label, children=children)
        self.feature_type = feature_type

class IntermediateNode(TreeNode):
    """
    Node representing the classes of the feature or threshold directions for the split.
    Should only ever have 1 child.
    """
    def __init__(self, label: str, children: List[TreeNode]=None):
        super().__init__(label=label, children=children)
        
class LeafNode(TreeNode):
    """
    Terminal node which should be labelled according to the decision_feature.
    """
    def __init__(self, label: str):
        super().__init__(label=label, children=[])

class DecisionTree:
    def __init__(self, train_data: pd.DataFrame, k: int, decision_feature: str, numerical_threshold_lim: int=2):
        """
        args:
            train_data = pandas dataframe
            numerical_threshold_lim: limit to how many numerical partitions can be made
                eg: lim = 2 means the age_set = [18, 19, 20, 20, 30, 50 ... ]
                    can be at most partitioned at 2 ages say <20 | >19
            k = random sample from K features
        """
        self.train_data = train_data
        self.categorical_features = ['Pclass', 'Sex', 'Embarked', 'Survived']
        self.numerical_threshold_lim = numerical_threshold_lim

        self.thresholder_types = {
            'median': self.median_thresholder,
            'mean': self.mean_thresholder,
            'iter': self.iterative_thresholder
        }
        
        if decision_feature not in train_data.columns:
            raise ValueError('Decision feature not in training data feature space')
        
        self.decision_feature = decision_feature

        # pre-process the decision_feature column for any NaN entries
        self.fill_nan_decision_feature()

        self.k = k

    def fill_nan_decision_feature(self) -> None:
        """
        Handles filling of NaN entries in decision feature column
        """
        # check the typing of entries of the decision_feature column
        if self.train_data[self.decision_feature].dtype == 'object' or self.decision_feature in self.categorical_features:

            # categorical features: fill NaN with mode of that column
            most_frequent_value = self.train_data[self.decision_feature].mode()[0]
            self.train_data[self.decision_feature] = self.train_data[self.decision_feature].fillna(most_frequent_value)
        else:
            # numerical features: fill NaN with median of that column
            mean_value = self.train_data[self.decision_feature].median()
            self.train_data[self.decision_feature].fillna(mean_value, inplace=True)

    def entropy(self, y: pd.Series) -> float:
        """
        Calculate the entropy of a series (target variable)
        """
        y = y.astype(str)
        values, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def info_entropy_categorical(self, data: pd.DataFrame, feature: str) -> float:
        """
        Returns weighted entropy for each subset based on categorical feature splitting.
        Treats NaN entries as separate category.
        """
        data[feature] = data[feature].fillna('NaN')

        subsets = data.groupby(feature)

        weighted_entropy = 0.0
        N = data.shape[0]

        # catch case if all data is NaN
        if N == 0:
            return 0

        for _, subset in subsets:
            target_values = subset[f'{self.decision_feature}']

            values, counts = np.unique(target_values, return_counts=True)
            probabilities = counts / target_values.size

            non_zero_probabilities = probabilities[probabilities > 0]

            subset_entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))

            N_subset = subset.shape[0]
            weighted_entropy += (N_subset / N) * subset_entropy

        return weighted_entropy

    def info_entropy_numerical(self, data: pd.DataFrame, feature: str, threshold: float) -> float:
        """
        Returns weighted entropy for each subset based on numerical threshold splitting.
        NaN entries are excluded from entropy calculation.
        """
        non_nan_data = data[~data[feature].isna()]

        subset_1 = non_nan_data[non_nan_data[feature] <= threshold]
        subset_2 = non_nan_data[non_nan_data[feature] > threshold]

        weighted_entropy = 0.0
        N = non_nan_data.shape[0]

        # catch case if all data is NaN
        if N == 0:
            return 0

        for subset in [subset_1, subset_2]:
            target_values = subset[f'{self.decision_feature}']

            values, counts = np.unique(target_values, return_counts=True)
            probabilities = counts / target_values.size

            non_zero_probabilities = probabilities[probabilities > 0]

            subset_entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))

            N_subset = subset.shape[0]
            weighted_entropy += (N_subset / N) * subset_entropy

        return weighted_entropy
    
    def info_gain_categorical(self, data: pd.DataFrame, feature: str) -> float:
        initial_entropy = self.entropy(data[f'{self.decision_feature}'])

        return initial_entropy - self.info_entropy_categorical(data, feature)
    
    def info_gain_numerical(self, data: pd.DataFrame, feature: str, threshold: float) -> float:
        initial_entropy = self.entropy(data[f'{self.decision_feature}'])

        return initial_entropy - self.info_entropy_numerical(data, feature, threshold)

    def iterative_thresholder(self, data: pd.DataFrame, feature: str) -> float:
        """
        Attempts a 2 bin partition at every middle point value.
        Selects split with maximal information gain.
        """
        # only want unique values to avoid trying the same split again
        sorted_values = data[feature].dropna().sort_values().unique()

        best_threshold = 0
        best_info_gain = -float('inf')

        for i in range(1, sorted_values.size):
            threshold = (sorted_values[i - 1] + sorted_values[i]) / 2
            
            info_gain = self.info_gain_numerical(data, feature, threshold)
            
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = threshold

        return best_threshold
    
    def median_thresholder(self, data: pd.DataFrame, feature: str) -> float:
        """
        Naively selects the median value of the numerical feature list as the threshold
        """
        return data[feature].median()

    def mean_thresholder(self, data: pd.DataFrame, feature: str) -> float:
        """
        Naively selects the mean value of the numerical feature list as threshold
        """
        return data[feature].mean()

    def id3(self, data: pd.DataFrame, k: int, thresholder: str) -> List:
        """
        Perform Iterative Dichotomiser 3 algorithm for splitting:
            1. Start at root
            2. Calculate info_gain (G) for subset k (of K) features
            3. Perform splitting based on maximal G
            4. Return splitting rule as encoded list to serve as node label
        """
        best_info_gain = -float('inf')

        # ensures we do not partition using the decision_feature data
        selection_features = [col for col in data.columns if col != f'{self.decision_feature}']

        # ensures valid k has been used
        k = min(k, len(selection_features))

        for feature in random.sample(selection_features, k):
            if feature in self.categorical_features:
                info_gain = self.info_gain_categorical(data, feature)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    splitting_feature = deque([feature])

            else:
                threshold = self.thresholder_types[thresholder](data, feature)

                info_gain = self.info_gain_numerical(data, feature, threshold)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    splitting_feature = deque([feature, threshold])

        if not splitting_feature:
            raise ValueError('No splitting has been formed.')

        # detect if a numerical or categorical splitting rule has been decided
        # append custom code for quick identification of which type of feature
        # 1 : numerical
        # 0 : categorical
        l = len(splitting_feature)
        if l == 2:
            splitting_feature.appendleft(1)
        elif l == 1:
            splitting_feature.appendleft(0)
        else:
            raise ValueError('Splitting feature list should not exceed length 2.')
        
        return splitting_feature

    def _learn(self, data: pd.DataFrame, k: int, thresholder: str, min_child_nodes: int=5) -> None:
        """
        Learn decision tree Recursively using ID3 until stopping conditions met.
        """
        # if for whatever reason we yield an empty split
        # we want learn algorithm to continue but flag the problem node
        if data.empty:
            return LeafNode(label='EMPTY')

        # if all data belongs to same decision feature class, return a LeafNode with that label
        if data[f'{self.decision_feature}'].nunique() == 1:
            return LeafNode(label=f"{data[f'{self.decision_feature}'].iloc[0]}")
        
        # if data smaller than threshold child node count to allow a split
        if data.shape[0] < min_child_nodes:
            return LeafNode(label=f"{data[f'{self.decision_feature}'].mode()[0]}")
        
        splitting_rule = self.id3(data, k, thresholder)
        
        # numerical splitting
        if splitting_rule[0] == 1:
            feature = splitting_rule[1]
            threshold = splitting_rule[2]

            # randomly allocate NaN entries
            nan_mask = data[feature].isna()
            nan_allocation = np.random.choice([True, False], size=nan_mask.sum())
            nan_indices = nan_mask[nan_mask].index
            nan_left_indices = nan_indices[nan_allocation]
            nan_right_indices = nan_indices[~nan_allocation]
            
            left_data = data[data[feature] <= threshold]
            right_data = data[data[feature] > threshold]

            left_data = pd.concat([left_data, data.loc[nan_left_indices]])
            right_data = pd.concat([right_data, data.loc[nan_right_indices]])

            parent_node = FeatureNode(label=f"{feature}", feature_type='num')

            # create left and right intermediate nodes
            left_node = IntermediateNode(label=f"<={threshold}")
            right_node = IntermediateNode(label=f">{threshold}")

            # recur into children nodes
            left_node.children = [self._learn(left_data, k, thresholder, min_child_nodes)]
            right_node.children = [self._learn(right_data, k, thresholder, min_child_nodes)]

            parent_node.children = [left_node, right_node]
            return parent_node

        # categorical splitting
        elif splitting_rule[0] == 0:
            feature = splitting_rule[1]
            data[feature] = data[feature].fillna("NaN")

            parent_node = FeatureNode(label=f"{feature}", feature_type='cat')
            children = []

            # iterate over categories
            for category, subset in data.groupby(feature):
                category_node = IntermediateNode(label=f"{category}")
                category_node.children = [self._learn(subset, k, thresholder, min_child_nodes)]
                children.append(category_node)

            parent_node.children = children
            return parent_node

        else:
            raise ValueError('Invalid splitting rule. Must be categorical (0) or numerical (1).')

    def learn(self, thresholder: str, min_child_nodes: int=5) -> None:
        self.root = self._learn(data=self.train_data, k=self.k, thresholder=thresholder, min_child_nodes=min_child_nodes)
    
    def decide(self, test_data: pd.DataFrame) -> List:
        """
        Traverse decision tree to evaluate prediction for each dataset row.
        Returns list of decisions
            decision_feature = Embarked returns ['Q','Q','C','S' ... ]
            decision_feature = Survived returns ['0','1','0','0' ... ]
        
        Args:
            test_data: pandas DataFrame of test data
        """
        decisions = []
        decision_count = 0

        for _, row in test_data.iterrows():
            current = self.root
            
            while isinstance(current, FeatureNode):
                # identify feature about which tree is splitting
                feature = current.label

                # numerical splitting case
                if current.feature_type == 'num':
                    threshold = float(current.children[0].label.split('<=')[-1])
                    if row[feature] <= threshold or pd.isna(row[feature]):
                        # if datum has NaN entry in this feature, also proceed to left child
                        current = current.children[0].children[0]
                    else:
                        current = current.children[1].children[0]

                # categorical splitting case
                elif current.feature_type == 'cat':
                    # check that test datum category was indeed seen in training
                    category_found = False
                    for child in current.children:
                        if child.label == str(row[feature]) or (pd.isna(row[feature]) and child.label == 'NaN'):
                            current = child.children[0]
                            category_found = True
                            break
                    
                    # if test datum has unseen category, proceed to random choice of child nodes
                    if not category_found:
                        n = len(child.children)
                        current = child.children[random.randint(0, n-1)]
            
            if isinstance(current, LeafNode):
                decision_count += 1
                if current.label == 'EMPTY':
                    # choose a random choice from the possible values of the decision feature category
                    random_decision = random.choice(self.decision_feature)
                    decisions.append(f"{random_decision}")
                else:
                    decisions.append(f"{current.label}")
            else:
                raise ValueError("Traversal ended on a non-leaf node.")

        return [decision_count, decisions]
    
    def print_tree(self) -> None:
        """
        Prints out decision tree by bfs traversal
        """
        if self.root is None:
            return

        queue = deque([(self.root, 0)])
        current_depth = 0
        current_level_nodes = []

        while queue:
            node, depth = queue.popleft()

            if depth > current_depth:
                # print current level nodes
                print("   ".join(current_level_nodes))
                current_level_nodes = []
                current_depth = depth
            
            # add current node to output list
            current_level_nodes.append(str(node.label))

            # queue the children
            if node.children:
                for child in node.children:
                    queue.append((child, depth + 1))
        
        if current_level_nodes:
            print("   ".join(current_level_nodes))


class RandomForest:
    def __init__(self, train_data: pd.DataFrame, decision_feature: str, k: int=6, tree_count: int=5):
        self.train_data = train_data
        self.decision_feature = decision_feature

        if tree_count % 2 == 0:
            tree_count += 1
            print('\n Have increased tree count by 1 to ensure odd number in forest.\n')

        # give each tree a bootstrap sample of the full training data set
        # IMPORTANT: make sure bootstrap samples keep original dataset unchanged

        copy_data = train_data.copy()
        self.forest = [
            DecisionTree(
                train_data=copy_data.sample(n=train_data.shape[0], replace=True).reset_index(drop=True),
                k=k,
                decision_feature=decision_feature
            )
            for _ in range(tree_count)
        ]

        # define features which need to be output as integer or string
        self.str_outs = ['Embarked', 'Sex']
        self.int_outs = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch']
        self.float_outs = ['Fare']

    def learn(self, thresholder: str, min_child_nodes: int=5) -> None:
        for tree in self.forest:
            tree.learn(thresholder=thresholder, min_child_nodes=min_child_nodes)

    def forest_vote(self, test_data: pd.DataFrame) -> List:
        tree_votes = []
        decision_count = set()

        for tree in self.forest:
            num_decisions, tree_decisions = tree.decide(test_data)
            decision_count.add(num_decisions)
            tree_votes.append(tree_decisions)

        # check that all entries of the tree_votes list are of the same length
        if len(decision_count) != 1:
            raise ValueError('Trees have returned decision arrays of differing lengths.')
        
        tree_votes_array = np.array(tree_votes)

        # compute majority vote over column direction
        majority_votes = []
        for column in tree_votes_array.T:
            vote_counts = Counter(column)

            # if all votes equally likely, choose the first one
            majority_vote = vote_counts.most_common(1)[0][0]
            majority_votes.append(str(majority_vote))

        # ensure the votes are converted to the correct data type
        if self.decision_feature in self.str_outs:
            majority_votes = [str(vote) for vote in majority_votes]
        elif self.decision_feature in self.int_outs:
            majority_votes = [int(vote) for vote in majority_votes]
        elif self.decision_feature in self.float_outs:
            majority_votes = [float(vote) for vote in majority_votes]
        else:
            raise ValueError(f"Unknown decision_feature type: {self.decision_feature}")

        return majority_votes
    
    def save_forest(self, folder: str, filename: str) -> None:
        """
        Save the trained forest to a file using pickle.
        """
        os.makedirs(folder, exist_ok=True)

        filepath = os.path.join(folder, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
