# decision_tree.py
# pure python decision tree algorithm using ID3 partitioning
import numpy as np
from typing import List
import pandas as pd
import random
from collections import deque, Counter
import pickle
import os
from itertools import combinations

class TreeNode:
    """
    Base class for decision tree nodes.
    """
    def __init__(self, label: str, children: List['TreeNode']|None=None):
        self.label = label
        self.children = children or []

class FeatureNode(TreeNode):
    """
    Node representing the feature about which the split is made.
    """
    def __init__(self, label: str, feature_type: str, children: List[TreeNode]|None=None):
        super().__init__(label=label, children=children)
        self.feature_type = feature_type

class IntermediateNode(TreeNode):
    """
    Node representing the classes of the feature or threshold directions for the split.
    Should only ever have 1 child.
    """
    def __init__(self, label: str, children: List[TreeNode]|None=None, parseable_bound: float=0.):
        super().__init__(label=label, children=children)
        self.parseable_bound = parseable_bound
        
class LeafNode(TreeNode):
    """
    Terminal node which should be labelled according to the decision_feature.
    """
    def __init__(self, label: str):
        super().__init__(label=label, children=[])

class DecisionTree:
    def __init__(self, train_data: pd.DataFrame, k: int, decision_feature: str, numerical_threshold_lim: int=1):
        """
        args:
            train_data = pandas dataframe
            numerical_threshold_lim: limit to how many numerical partitions can be made
                eg: lim = 2 means the age_set = [18, 19, 20, 20, 30, 50 ... ]
                    can be at most partitioned at 2 ages say <20 | >19
            k = random sample from K features
        """
        self.train_data = train_data
        self.categorical_features = ['Location']
        self.numerical_threshold_lim = numerical_threshold_lim

        self.thresholder_types = {
            'median': self.median_thresholder,
            'mean': self.mean_thresholder,
            'iter_bf': self.iterative_thresholder_bf
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
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return float(-np.sum(probabilities * np.log2(probabilities)))

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
            target_values = subset[self.decision_feature]

            _, counts = np.unique(target_values, return_counts=True)
            probabilities = counts / target_values.size

            non_zero_probabilities = probabilities[probabilities > 0]

            subset_entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))

            N_subset = subset.shape[0]
            weighted_entropy += (N_subset / N) * subset_entropy

        return float(weighted_entropy)

    def info_entropy_numerical(self, data: pd.DataFrame, feature: str, thresholds: List[float]) -> float:
        """
        Returns weighted entropy for each subset based on numerical threshold splitting.
        NaN entries are excluded from entropy calculation.
        """
        non_nan_data = data[~data[feature].isna()]

        subsets = []
        n = len(thresholds)

        if n == 1:
            x = thresholds[0]
            subsets.append(non_nan_data[non_nan_data[feature] <= x])
            subsets.append(non_nan_data[non_nan_data[feature] > x])
        else:
            for i, x in enumerate(thresholds):
                if i == 0:
                    subsets.append(non_nan_data[non_nan_data[feature] <= x])
                if i == n-1:
                    y = thresholds[i-1]
                    subsets.append(non_nan_data[(non_nan_data[feature] > y) & (non_nan_data[feature] <= x)])
                    subsets.append(non_nan_data[non_nan_data[feature] > x])
                else:
                    y = thresholds[i-1]
                    subsets.append(non_nan_data[(non_nan_data[feature] > y) & (non_nan_data[feature] <= x)])
            
        weighted_entropy = 0.0
        N = non_nan_data.shape[0]

        # catch case if all data is NaN
        if N == 0:
            return 0

        for subset in subsets:
            target_values = subset[self.decision_feature]

            _, counts = np.unique(target_values, return_counts=True)
            probabilities = counts / target_values.size

            non_zero_probabilities = probabilities[probabilities > 0]

            subset_entropy = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities))

            N_subset = subset.shape[0]
            weighted_entropy += (N_subset / N) * subset_entropy

        return weighted_entropy
    
    def info_gain_categorical(self, data: pd.DataFrame, feature: str) -> float:
        initial_entropy = self.entropy(data[self.decision_feature])

        return initial_entropy - self.info_entropy_categorical(data, feature)
    
    def info_gain_numerical(self, data: pd.DataFrame, feature: str, thresholds: List[float]) -> float:
        initial_entropy = self.entropy(data[self.decision_feature])

        return initial_entropy - self.info_entropy_numerical(data, feature, thresholds)
    
    def iterative_thresholder_bf(self, data: pd.DataFrame, feature: str) -> List[float]:
        """
        Brute Force algorithm.
        Attempts splitting data into multiple bins by choosing randomly selected split points.
        Selects best split from random selection.
        
        args:
            data: dataset for splitting
            feature: feature to dictate entropies
            num_thresholds: number of thresholds (bins = thresholds + 1)
        """
        # only want unique values to avoid trying the same split again
        sorted_values = data[feature].dropna().sort_values().unique()

        num_thresholds = self.numerical_threshold_lim
        # catch for cases when the threshold count exceeds the data size
        if sorted_values.size < self.numerical_threshold_lim:
            num_thresholds = 1

        best_thresholds = []
        best_info_gain = -float('inf')

        # generate all possible combs of thresholds
        for indices in combinations(range(1, sorted_values.size), num_thresholds):
            thresholds = [(sorted_values[i - 1] + sorted_values[i]) / 2 for i in indices]

            info_gain = self.info_gain_numerical(data, feature, thresholds)

            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_thresholds = thresholds

        return best_thresholds

    def median_thresholder(self, data: pd.DataFrame, feature: str) -> List[float]:
        """
        Naively selects the median value of the numerical feature list as the threshold
        """
        # return as list for compatibility with iterative method
        return [data[feature].median()]

    def mean_thresholder(self, data: pd.DataFrame, feature: str) -> List[float]:
        """
        Naively selects the mean value of the numerical feature list as threshold
        """
        # return as list for compatibility with iterative method
        return [data[feature].mean()]

    def id3(self, data: pd.DataFrame, k: int, thresholder: str) -> tuple[int, str, list[float]|None]:
        """
        Perform Iterative Dichotomiser 3 algorithm for splitting:
            1. Start at root
            2. Calculate info_gain (G) for subset k (of K) features
            3. Perform splitting based on maximal G
            4. Return splitting rule as encoded list to serve as node label
        """

        # ensures we do not partition using the decision_feature data
        selection_features = [col for col in data.columns if col != self.decision_feature]

        # ensures valid k has been used
        k = min(k, len(selection_features))

        best_info_gain = -float('inf')
        splitting_feature = None
        for feature in random.sample(selection_features, k):
            if feature in self.categorical_features:
                info_gain = self.info_gain_categorical(data, feature)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    splitting_feature = (feature, None)

            else:
                thresholds = self.thresholder_types[thresholder](data, feature)

                info_gain = self.info_gain_numerical(data, feature, thresholds)

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    splitting_feature = (feature, thresholds)

        if splitting_feature is None:
            raise ValueError('No splitting has been formed.')

        # detect if a numerical or categorical splitting rule has been decided
        # append custom code for quick identification of which type of feature
        # 1 : numerical
        # 0 : categorical
        feat, thresh = splitting_feature

        return (0 if thresh is None else 1, feat, thresh)

    def _learn(self, data: pd.DataFrame, k: int, thresholder: str, min_child_nodes: int=5) -> TreeNode:
        """
        Learn decision tree Recursively using ID3 until stopping conditions met.
        """
        # if for whatever reason we yield an empty split
        # we want learn algorithm to continue but flag the problem node
        if data.empty:
            return LeafNode(label='EMPTY')

        # if all data belongs to same decision feature class, return a LeafNode with that label
        if data[f'{self.decision_feature}'].nunique() == 1:
            return LeafNode(label=f"{data[self.decision_feature].iloc[0]}")
        
        # if data smaller than threshold child node count to allow a split
        if data.shape[0] < min_child_nodes:
            return LeafNode(label=f"{data[self.decision_feature].mode()[0]}")
        
        splitting_rule = self.id3(data, k, thresholder)
        # print(splitting_rule)
        # print(data)
        
        # numerical splitting
        if splitting_rule[0] == 1:
            feature = splitting_rule[1]
            thresholds = splitting_rule[2]

            # assert thresholds, 'Thresholds list is empty.'
            if not thresholds:
                return LeafNode(label=f"{data[self.decision_feature].mode()[0]}")

            # in case thresholds not sorted
            thresholds = sorted(thresholds)

            # filter NaN entries to randomly allocate
            nan_mask = data[feature].isna()
            nan_allocation = np.random.choice(range(len(thresholds) + 1), size=nan_mask.sum())
            nan_indices = nan_mask[nan_mask].index

            # create subsets based on thresholds
            subsets = []
            previous_threshold = None

            for i, threshold in enumerate(thresholds):
                if i == 0:
                    subset = data[data[feature] <= threshold]
                else:
                    subset = data[(data[feature] > previous_threshold) & (data[feature] <= threshold)]

                # add NaN values designated to this bin
                nan_subset_indices = nan_indices[nan_allocation == i]
                subset = pd.concat([subset, data.loc[nan_subset_indices]])
                subsets.append(subset)

                previous_threshold = threshold

            # handle values > final threshold
            final_subset = data[data[feature] > thresholds[-1]]
            nan_subset_indices = nan_indices[nan_allocation == len(thresholds)]
            final_subset = pd.concat([final_subset, data.loc[nan_subset_indices]])
            subsets.append(final_subset)

            parent_node = FeatureNode(label=f"{feature}", feature_type='num')

            category_nodes = []

            # create label for category node(s)
            for i, subset in enumerate(subsets):
                if i == 0:
                    label = f"<= {thresholds[0]}"
                    parseable_bound = thresholds[0]

                elif i == len(thresholds):
                    label = f"> {thresholds[-1]}"
                    parseable_bound = thresholds[-1]
                else:
                    label = f"> {thresholds[i-1]} & <= {thresholds[i]}"
                    parseable_bound = thresholds[i]

                category_node = IntermediateNode(label=label, parseable_bound=parseable_bound)
                category_node.children = [self._learn(subset, k, thresholder, min_child_nodes)]
                category_nodes.append(category_node)

            parent_node.children = category_nodes
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
                    thresholds = [category_node.parseable_bound for category_node in current.children[:-1]]

                    feature_value = row[feature]

                    if pd.isna(feature_value):
                        # NaN values go to last child by default
                        current = current.children[-1].children[0]
                    else:
                        # find appropiate threshold range
                        for i, threshold in enumerate(thresholds):
                            if feature_value <= threshold:
                                current = current.children[i].children[0]
                                break
                        else:
                            # feature_value > all thresholds, proceed to last child node
                            current = current.children[-1].children[0]

                # categorical splitting case
                elif current.feature_type == 'cat':
                    # check that test datum category was indeed seen in training
                    category_found = False
                    child = None
                    for child in current.children:
                        if child.label == str(row[feature]) or (pd.isna(row[feature]) and child.label == 'NaN'):
                            current = child.children[0]
                            category_found = True
                            break
                    
                    # if test datum has unseen category, proceed to random choice of child nodes
                    if not category_found:
                        assert child is not None, 'No child node found for category.'
                        current = random.choice(child.children)
            
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
    def __init__(self, train_data: pd.DataFrame, decision_feature: str, k: int=6, tree_count: int=5, numerical_threshold_lim: int=1):
        """
        args:
            train_data: dataset to train decision trees on
            decision_feature: feature to learn for
            k: number of features to sample from in id3 algo
            tree_count: number of trees in forest (odd)
            numerical_threshold_lim: number of numerical thresholds up to which we search for (bins = thresholds + 1)
        """
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
                decision_feature=decision_feature,
                numerical_threshold_lim=numerical_threshold_lim
            )
            for _ in range(tree_count)
        ]

        # define features which need to be output as integer or string
        self.str_outs = ['Location']
        self.int_outs = []
        self.float_outs = train_data.columns[1:].tolist()

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
