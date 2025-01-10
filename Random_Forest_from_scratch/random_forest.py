# playground for random forest
from decision_tree import RandomForest, DecisionTree
import pandas as pd
import numpy as np
from collections import deque
from typing import List
from cross_val import CrossValidation
import pickle
import os

def load_forest(folder: str, filename: str) -> RandomForest:
    """
    Load a trained RandomForest from a pickle file.
    """
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")
    
    with open(filepath, 'rb') as f:
        forest = pickle.load(f)

    return forest

def cross_validation(decision_feature: str, hyperparams: List[int], k: int=5) -> List[float]:
    """
    Run k-fold cross-validation.
    
    Args:
        hyperparams: list of tree_counts to validate
        k: number of cross-validation folds

    eg:
        hyperparams = [3, 5, 7] (forests of size 3, 5, 7)
        k = 5 (default)

        1. forests are each trained on 60% of the k-1 folds
        2. best hyperparameter (by accuracy) selected by validation on remaining 40%
        3. optimal hyperparameter evaluated on k-1^th fold (unseen)
    """
    cross_val = CrossValidation(data, k)
    splits = cross_val.get_splits()

    # score of best forest on each test fold
    split_scores = []

    for i, split in enumerate(splits):
        print(f"\nProcessing fold {i + 1} of {k}...")
        train_data, validation_data, test_data = split

        # score of the best forest, after validation selection
        best_forest_score = 0

        for tree_count in hyperparams:
            forest = RandomForest(
                train_data=train_data,
                decision_feature=decision_feature,
                tree_count=tree_count
                )
            
            forest.learn(thresholder='iter')

            # create prediction array
            forest_prediction = forest.forest_vote(validation_data)

            # calculate forest accuracy
            true_values = validation_data[decision_feature].values
            correct_predictions = sum(
                pred == true for pred, true in zip(forest_prediction, true_values)
            )
            accuracy = correct_predictions / len(true_values)

            print(f"Fold {i + 1} forest with {tree_count} trees accuracy: {accuracy}")

            # filter for best forest within hyperparameter set
            if accuracy > best_forest_score:
                best_forest_score = accuracy
                best_forest = f'fold={i + 1}_h={tree_count}'

                forest.save_forest(folder='forests_folder', filename=f'fold={i + 1}_h={tree_count}')

        validated_forest = load_forest(folder='forests_folder', filename=best_forest)

        validated_forest_prediction = validated_forest.forest_vote(test_data)

        true_values = test_data[decision_feature].values
        correct_predictions = sum(
            pred == true for pred, true in zip(validated_forest_prediction, true_values)
        )
        accuracy = correct_predictions / len(true_values)

        split_scores.append(float(np.round(accuracy, 3)))

    # flush the forests_folder
    for file in os.listdir('forests_folder'):
            file_path = os.path.join('forests_folder', file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return split_scores

            
selected_features = 'Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked'
data = pd.read_csv('Random_Forest_from_scratch/titanic/train.csv')[selected_features.split(',')]

final_scores = cross_validation(decision_feature='Pclass', hyperparams=[3,5,7])

print(f"\n Scores for best forest after validation on each test fold: {final_scores}")













