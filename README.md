__Projects:__

1. Random Forest:
   Implemented with just NumPy & Pandas. Utlisises ID3 with Information Entropy (rather than Gini Impurity) to generate optimal partitions.
   Supports continuous data which is binned into leaves representing discrete ranges. Also supports categorical data. Was tested with the Kaggle
   Titanic dataset and a survey of Danish waste statisticcs per district. Can train individual DecisionTree objects with method to print out
   the learned tree structure for interpreting.

2. MLP:
   Simple feed forward network in pure NumPy supporting various activation functions, batched training, learning rate decay and various optimisers (GD, SGD, Adam
   & Lion). Trained as a toy model on simple non-linear functions.

3. PINN:
   Building from the MLP backbone still (stubbornly) in pure NumPy this currently supports a hardcoded autograd to retrieve network derivatives and 2nd derivatives
   with respect to model inputs (Jacobian & diagonal Hessian) to construct differential Physics Loss functions. Finite difference methods for approximating the model
   derivatives are also supported (used in testing to validate the manual autograd propagation). Currently implementing batch normalisation layers to help regularisation
   problems to do with vanishing/diverging network gradients.
