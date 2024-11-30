import numpy as np
from _criterion import *

CRITERIA_CLF = {
    "gini": ClassificationCriterion.gini,
    "log_loss": ClassificationCriterion.log_loss,
    "entropy": ClassificationCriterion.entropy
}

class Tree:
    """Base Decision Tree Node"""
    def __init__(
        self,
        feature=None,
        threshold=None,
        value=None,
        impurity=None,
        children_left=None,
        children_right=None,
        is_leaf=False,
        n_sample=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.impurity = impurity
        self.children_left = children_left
        self.children_right = children_right
        self.is_leaf = is_leaf
        self.n_sample = n_sample

class BaseDecisionTree:
    """Base Decision Tree"""
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        alpha=0.0, # cost-complexity pruning
        max_features=None
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.alpha = alpha
        self.tree_ = None
        self.max_features = max_features 
        
        self._impurity_evaluation = None
        self._leaf_value_calculation = None

    def fit(self, X, y):
        """
        Build a decision tree.
        """
        # Convert the input to numpy arrays
        X = np.array(X).copy()
        y = np.array(y).copy()

        # Extract the data shape
        self.n_samples, self.n_features = X.shape

        # Grow the tree
        self.tree_ = self._grow_tree(X, y)

    def _calculate_impurity_decrease(self, X, y, feature_i, threshold):
        """Calculate the impurity decrease after splitting at feature_i and threshold."""
        parent_impurity = self._impurity_evaluation(y)

        # Split the data based on feature_i and threshold
        left_mask = X[:, feature_i] < threshold
        right_mask = ~left_mask

        y_left = y[left_mask]
        y_right = y[right_mask]

        left_impurity = self._impurity_evaluation(y_left)
        right_impurity = self._impurity_evaluation(y_right)

        n_left = len(y_left)
        n_right = len(y_right)
        n_total = len(y)

        weighted_child_impurity = (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity
        impurity_decrease = parent_impurity - weighted_child_impurity

        return impurity_decrease

    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on based on impurity decrease.
        """
        best_impurity_decrease = -float('inf')
        best_feature_i = None
        best_threshold = None

        n_features = X.shape[1]

        # Determine the number of features to consider based on max_features
        if self.max_features is not None:
            if isinstance(self.max_features, int):
                features_to_consider = np.random.choice(n_features, self.max_features, replace=False)
            elif isinstance(self.max_features, float):
                # Fraction of features to consider
                num_features = max(1, int(self.max_features * n_features))
                features_to_consider = np.random.choice(n_features, num_features, replace=False)
            elif self.max_features == "sqrt":
                # sqrt(n_features)
                num_features = int(np.sqrt(n_features))
                features_to_consider = np.random.choice(n_features, num_features, replace=False)
            elif self.max_features == "log2":
                # log2(n_features)
                num_features = int(np.log2(n_features))
                features_to_consider = np.random.choice(n_features, num_features, replace=False)
            else:
                raise ValueError(f"Invalid value for max_features: {self.max_features}")
        else:
            # Consider all features
            features_to_consider = range(n_features)

        # Iterate over the selected features
        for feature_i in features_to_consider:
            thresholds = np.unique(X[:, feature_i])

            # Try splitting at each threshold
            for threshold in thresholds:
                impurity_decrease = self._calculate_impurity_decrease(X, y, feature_i, threshold)

                # If the split is better, update the best split
                if impurity_decrease > best_impurity_decrease:
                    best_impurity_decrease = impurity_decrease
                    best_feature_i = feature_i
                    best_threshold = threshold

        return best_feature_i, best_threshold
    
    def _grow_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""

        # Base case: if the number of samples is less than min_samples_split or no valid split can be found
        if len(y) <= self.min_samples_split or len(np.unique(y)) == 1:
            node_value = self._leaf_value_calculation(y)  # Calculate the majority vote or leaf value
            return Tree(
                value=node_value,
                is_leaf=True,
                n_sample=len(y),
            )

        # Create node
        node_impurity = self._impurity_evaluation(y)  # Calculate impurity
        node_value = self._impurity_evaluation(y)  # Result of predict
        node = Tree(
            impurity=node_impurity,
            value=node_value,
            is_leaf=False,  # This is not a leaf yet
            n_sample=self.n_samples,
        )

        # Split recursively only if we are allowed to grow the tree
        if self.max_depth is not None:
            cond = depth < self.max_depth  # Only compare depth if max_depth is defined
        else:
            cond = True  # If max_depth is None, continue growing indefinitely

        if cond:
            # Find the best split
            feature_i, threshold = self._best_split(X, y)

            # If a valid split was found
            if feature_i is not None:
                left_mask = X[:, feature_i] < threshold
                right_mask = ~left_mask

                # Apply masks to split data
                X_left = X[left_mask]
                X_right = X[right_mask]
                y_left = y[left_mask]
                y_right = y[right_mask]

                # If either left or right side has fewer than min_samples_leaf, stop recursion
                if len(X_left) <= self.min_samples_leaf or len(X_right) <= self.min_samples_leaf:
                    return Tree(
                        value=self._leaf_value_calculation(y),  # Majority vote or leaf value
                        is_leaf=True,
                        n_sample=len(y),
                    )

                # Recursive calls to grow the tree
                node.feature = feature_i
                node.threshold = threshold
                node.children_left = self._grow_tree(X_left, y_left, depth + 1)
                node.children_right = self._grow_tree(X_right, y_right, depth + 1)

        return node
        
    def _predict_value(self, sample):
        """Recursively predict the value for a single sample."""
        node = self.tree_  # Start from the root of the tree
        while not node.is_leaf:  # Keep going until a leaf node is reached
            if sample[0, node.feature] < node.threshold:
                node = node.children_left
            else:
                node = node.children_right
        return node.value  # Return the value at the leaf node

    def predict(self, X):
        """Predict the target values for a given dataset X."""
        X = np.array(X).copy()

        # Predict
        y = [self._predict_value(sample.reshape(1, -1)) for sample in X]

        return np.array(y)
        
class DecisionTreeClassifier(BaseDecisionTree):
    """Decision Tree Classifier (inherits from BaseDecisionTree)."""
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        alpha=0.0,
        max_features=None
    ):
        super().__init__(  
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_impurity_decrease=min_impurity_decrease,
            alpha=alpha,
            max_features=max_features
        )
    
    def _impurity_evaluation(self, y):
        """Impurity evaluation based on the selected criterion."""
        if self.criterion == "gini":
            return ClassificationCriterion.gini(y)
        elif self.criterion == "log_loss":
            return ClassificationCriterion.log_loss(y)
        elif self.criterion == "entropy":
            return ClassificationCriterion.entropy(y)
        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")

    def _calculate_majority_vote(self, y):
        """Calculate the majority vote for classification."""
        vals, counts = np.unique(y, return_counts=True)
        
        # Find majority
        ind_max = np.argmax(counts)
        y_pred = vals[ind_max]
        
        return y_pred
    
    def fit(self, X, y):
        self._impurity_evaluation = CRITERIA_CLF[self.criterion]
        self._leaf_value_calculation = self._calculate_majority_vote
        
        super().fit(X, y)  # Call fit method of the base class
