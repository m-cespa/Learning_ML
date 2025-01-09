# decision_tree.py
# pure python decision tree algorithm using ID3 partitioning
import numpy as np
from typing import List
import pandas as pd
import random
from collections import deque

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
    Terminal node which should be labelled:
        Survived=Y
        Survived=N
    """
    def __init__(self, label: str):
        super().__init__(label=label, children=[])

class DecisionTree:
    def __init__(self, train_data: pd.DataFrame, k: int, numerical_threshold_lim: int=2):
        """
        args:
            train_data = pandas dataframe
            numerical_threshold_lim: limit to how many numerical partitions can be made
                eg: lim = 2 means the age_set = [18, 19, 20, 20, 30, 50 ... ]
                    can be at most partitioned at 2 ages say <20 | >19
            k = random sample from K features
        """
        self.train_data = train_data
        self.categorical_features = ['Pclass', 'Sex', 'Embarked']
        self.numerical_threshold_lim = numerical_threshold_lim

        self.thresholder_types = {
            'median': self.median_thresholder,
            'mean': self.mean_thresholder,
            'iter': self.iterative_thresholder
        }
        
        self.k = k

    def entropy(self, y: pd.Series) -> float:
        """
        Calculate the entropy of a series (target variable)
        """
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

        for _, subset in subsets:
            target_values = subset['Survived']

            values, counts = np.unique(target_values, return_counts=True)
            probabilities = counts / target_values.size
            subset_entropy = -np.sum(probabilities * np.log2(probabilities))

            N_subset = subset.shape[0]
            weighted_entropy += (N_subset / N) * subset_entropy

        return weighted_entropy

    def info_entropy_numerical(self, data: pd.DataFrame, feature: str, threshold: float) -> float:
        """
        Returns weighted entropy for each subset based on numerical threshold splitting.
        Randomly allocates NaN entries into one of the two subsets.
        """
        nan_mask = data[feature].isna()

        nan_allocation = np.random.choice([True, False], size=nan_mask.sum())
        nan_indices = nan_mask[nan_mask].index

        subset_1_mask = (data[feature] <= threshold) | (data.index.isin(nan_indices[nan_allocation]))
        subset_2_mask = (data[feature] > threshold) | (data.index.isin(nan_indices[~nan_allocation]))

        subset_1 = data[subset_1_mask]
        subset_2 = data[subset_2_mask]

        weighted_entropy = 0.0
        N = data.shape[0]

        for subset in [subset_1, subset_2]:
            target_values = subset['Survived']

            values, counts = np.unique(target_values, return_counts=True)
            probabilities = counts / target_values.size
            subset_entropy = -np.sum(probabilities * np.log2(probabilities))

            N_subset = subset.shape[0]
            weighted_entropy += (N_subset / N) * subset_entropy

        return weighted_entropy
    
    def info_gain_categorical(self, data: pd.DataFrame, feature: str) -> float:
        initial_entropy = self.entropy(data['Survived'])

        return initial_entropy - self.info_entropy_categorical(data, feature)
    
    def info_gain_numerical(self, data: pd.DataFrame, feature: str, threshold: float) -> float:
        initial_entropy = self.entropy(data['Survived'])

        return initial_entropy - self.info_entropy_numerical(data, feature, threshold)

    def iterative_thresholder(self, data: pd.DataFrame, feature: str) -> float:
        """
        Attempts a 2 bin partition at every middle point value.
        Selects split with maximal information gain.
        """
        # only want unique values to avoid trying the same split again
        sorted_values = data[feature].dropna().sort_values().unique()

        best_threshold = None
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
        sorted_values = data[feature].sort_values().values
        n = sorted_values.size

        if n % 2 == 0:
            return 0.5 * (sorted_values[n//2 - 1] + sorted_values[n//2])
        else:
            return sorted_values[n//2]

    def mean_thresholder(self, data: pd.DataFrame, feature: str) -> float:
        """
        Naively selects the mean value of the numerical feature list as threshold
        """
        target_values = data[feature].values
        return np.nanmean(target_values)

    def id3(self, data: pd.DataFrame, k: int, thresholder: str) -> List:
        """
        Perform Iterative Dichotomiser 3 algorithm for splitting:
            1. Start at root
            2. Calculate info_gain (G) for subset k (of K) features
            3. Perform splitting based on maximal G
            4. Return splitting rule as encoded list to serve as node label
        """
        if k > data.shape[1]:
            raise ValueError('k cannot be larger than total feature count.')
        
        best_info_gain = -float('inf')

        # ensures we do not partition using the Survived (label) data
        selection_features = [col for col in data.columns if col != 'Survived']

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

    def _learn(self, data: pd.DataFrame, k: int, thresholder: str, min_child_nodes: int=10) -> TreeNode:
        """
        Learn decision tree Recursively using ID3 until stopping conditions met.
        """
        # if for whatever reason we yield an empty split
        # we want learn algorithm to continue but flag the problem node
        if data.empty:
            return LeafNode(label='EMPTY')

        # if all data belongs to same 'Survived' class, return that value (1 or 0)
        if data['Survived'].nunique() == 1:
            return LeafNode(label=f"Survived:{data['Survived'].iloc[0]}")
        
        # if data smaller than threshold child node count to allow a split
        if data.shape[0] < min_child_nodes:
            return LeafNode(label=f"Survived:{data['Survived'].mode()[0]}")
        
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

    def learn(self, thresholder: str) -> TreeNode:
        self.root = self._learn(data=self.train_data, k=self.k, thresholder=thresholder)
        return self.root
    
    def decide(self, test_data: pd.DataFrame) -> bool:
        current = self.root

        while isinstance(current, FeatureNode):
            feature = current.label

            if feature in test_data.columns:
                feature_value = test_data[feature].iloc[0]

                # categorical split
                if current.feature_type == 'cat':
                    if isinstance(current.children[0], IntermediateNode):
                        for category_node in current.children:
                            if category_node.label == feature_value:
                                # once we find the match, move to the child
                                # IntermediateNode should only have 1 child
                                current = category_node.children[0]

                # numerical split
                elif current.feature_type == 'num':
                    if isinstance(current.children[0], IntermediateNode):
                        # this tree uses convention that left child is always the <= case
                        threshold_node = current.children[0]
                        threshold = float(threshold_node.label.split('=')[1])

                        if feature_value <= threshold:
                            # go to left child
                            current = threshold_node.children[0]
                        else:
                            # go to right child
                            current = threshold_node.children[1]

            # if feature does not exist in test_data columns
            # arbitrarily move to down 2 steps to the left
            else:
                current = current.children[0].children

        # terminal case
        if isinstance(current, LeafNode):
            if current.label == 'Survived:1':
                return True
            else:
                return False
        else:
            raise ValueError("Reached a potentially terminal node which wasn't a leaf?")
