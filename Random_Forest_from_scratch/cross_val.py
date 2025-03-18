from sklearn.model_selection import KFold
from typing import List, Tuple
import pandas as pd

class CrossValidation:
    def __init__(self, data: pd.DataFrame, k: int=5, random_state: int=42):
        """
        Wrapper for k-fold cross-validation
        """
        # shuffle the dataframe when loading
        self.data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)
        self.k = k
        self.random_state = random_state
        self.kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)

    def get_splits(self) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Split the dataset into k folds and generate data tuple for each fold.

        returns:
            splits = [tuple_1, tuple_2 ... ]
        """
        splits = []
        
        for rest_index, test_index in self.kfold.split(self.data):
            test_data = self.data.iloc[test_index]
            rest_data = self.data.iloc[rest_index]
            
            # of the k-1 train+validate folds, allocate 60% to training | 40% to validation
            train_size = int(0.6 * len(rest_data))
            train_data = rest_data.iloc[:train_size]
            validation_data = rest_data.iloc[train_size:]
            
            splits.append((train_data, validation_data, test_data))
        
        return splits
