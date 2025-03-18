# playground for random forest
from decision_tree import RandomForest, DecisionTree
import pandas as pd
import numpy as np
from collections import deque
from typing import List
from cross_val import CrossValidation
import pickle
import os
import matplotlib.pyplot as plt
import ast

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

def plot_cross_val_scores(all_scores: List, hyperparams: List[int]):
    all_scores = np.array(all_scores)

    num_folds = all_scores.shape[0]
    
    plt.figure(figsize=(10, 6))
    
    for i, hyperparam in enumerate(hyperparams):
        plt.plot(range(1, num_folds + 1), all_scores[:, i], marker='o', label=f'{hyperparam} trees')

    plt.title('Cross-validation Scores for Different Hyperparameters', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(range(1, num_folds + 1))
    plt.legend(title='Tree count')
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def cross_validation(data: pd.DataFrame, decision_feature: str, hyperparams: List[int], feature_sample_count: int, k: int=5, min_child_nodes: int=5, numerical_threshold_lim: int=1) -> List:
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

    # all scores for each hyperparameter at each fold
    all_scores = []

    for i, split in enumerate(splits):
        print(f"\nProcessing fold {i + 1} of {k}...")
        train_data, validation_data, test_data = split

        # keeps track of which hyperparameter forest scores best in validation
        best_forest_score = 0
        fold_scores = []

        for tree_count in hyperparams:
            forest = RandomForest(
                train_data=train_data,
                decision_feature=decision_feature,
                tree_count=tree_count,
                k=feature_sample_count,
                numerical_threshold_lim=numerical_threshold_lim
                )
            
            forest.learn(thresholder='iter_bf', min_child_nodes=min_child_nodes)

            # create prediction array
            forest_prediction = forest.forest_vote(validation_data)

            # print(forest_prediction)

            # calculate forest accuracy
            true_values = validation_data[decision_feature].values
            # correct_predictions = sum(
            #     pred == true for pred, true in zip(forest_prediction, true_values)
            # )
            correct_predictions_validation = 0

            for pred, true in zip(forest_prediction, true_values):
                try:
                    # Convert the string bound into a list of numbers
                    lower, upper = ast.literal_eval(pred)  # Convert '[1000,2000]' -> [1000, 2000]
                    
                    # Check if true value falls within the predicted bounds
                    if lower <= true <= upper:
                        correct_predictions_validation += 1
                except (ValueError, SyntaxError, TypeError):
                    # If parsing fails, fall back to direct comparison (for categorical predictions)
                    if pred == true:
                        correct_predictions_validation += 1

            accuracy = correct_predictions_validation / len(true_values)

            fold_scores.append(accuracy)

            print(f"Fold {i + 1} forest with {tree_count} trees accuracy: {accuracy}")

            # filter for best forest within hyperparameter set
            if accuracy > best_forest_score:
                best_forest_score = accuracy
                best_forest = f'fold={i + 1}_h={tree_count}'

                forest.save_forest(folder='forests_folder', filename=f'fold={i + 1}_h={tree_count}')

        all_scores.append(fold_scores)

        validated_forest = load_forest(folder='forests_folder', filename=best_forest)

        validated_forest_prediction = validated_forest.forest_vote(test_data)

        true_values = test_data[decision_feature].values

        correct_predictions_testing = 0
        # correct_predictions = sum(
        #     pred == true for pred, true in zip(validated_forest_prediction, true_values)
        # )

        for pred, true in zip(validated_forest_prediction, true_values):
            try:
                lower, upper = ast.literal_eval(pred)
                if lower <= true <= upper:
                    correct_predictions_testing += 1
            except (ValueError, SyntaxError, TypeError):
                if pred == true:
                    correct_predictions_testing += 1

        accuracy = correct_predictions_testing / len(true_values)

        split_scores.append(float(np.round(accuracy, 3)))

    # flush the forests_folder
    for file in os.listdir('forests_folder'):
            file_path = os.path.join('forests_folder', file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return split_scores, all_scores

            
# selected_features = 'Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked'
data = pd.read_csv('Random_Forest_from_scratch/denmark_waste/2013_data_totalwaste.csv').drop(['Location'], axis=1)

# removing All_denmark and Copenhagen outliers
train_data = data.iloc[2:].copy()

final_scores, all_scores = cross_validation(
                                            data=train_data,
                                            decision_feature='TOTAL HOUSEHOLD WASTE', 
                                            hyperparams=[3, 5, 7], 
                                            k=5, 
                                            min_child_nodes=8,
                                            feature_sample_count=17,
                                            numerical_threshold_lim=1)

print(f"\n Scores for best forest after validation on each test fold: {final_scores}")

plot_cross_val_scores(all_scores, hyperparams=[3, 5, 7])
