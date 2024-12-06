import numpy as np
import pandas as pd

class DecisionTreeClassifierCostSensitivity:
    """
    Decision Tree Classifier with cost sensitivity.

    Parameters:
    -----------
    - criterion : str, default='gini'
        The function to measure the quality of a split. Supported criteria are 'gini' for Gini impurity
        and 'log_loss' for logarithmic loss.

    - splitter : str, default='best'
        The strategy used to choose the split at each node. Supported strategies are 'best' to choose
        the best split and 'random' to choose the best random split.

    - max_features : int, float, str, or None, default=None
        - int: The number of features to consider.
        - float: The fraction of features to consider.
        - 'sqrt': The square root of the number of features.
        - 'log2': The base-2 logarithm of the number of features.
        - None: Use all features.

    - max_depth : int or None, default=None
        The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or
        until all leaves contain less than min_samples_split samples.

    - min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - int: The minimum number of samples.
        - float: The fraction of samples.

    - min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node:
        - int: The minimum number of samples.
        - float: The fraction of samples.

    - cost_matrix : np.ndarray, default=None
        A matrix where C[i, j] represents the cost of misclassifying an instance of class i as class j.

    fitting model:
    -----------
    model = DecisionTreeClassifierCostSensitivity()
    model.fit(X, y) --> if not using cost matrix
    model.fit(X, y, cost_matrix) --> if using cost matrix
        - cost_matrix = np.array([[0, 10],
                                 [5, 0]])

    predict:
    --------
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    """
    def __init__(self, criterion='gini', splitter='best', max_features=None, 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Initialize the model with given hyperparameters.
        """
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y, cost_matrix=None):
        """
        Fit the model to the training data.

        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            The training data.

        y : pd.Series or np.ndarray
            The target values.

        cost_matrix : np.ndarray, default=None
            A matrix where C[i, j] represents the cost of misclassifying an instance of class i as class j.
        """
        # Ensure X and y are numpy arrays for consistency
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y = y.values if isinstance(y, pd.Series) else y

        # Check if the length of X and y are equal
        if len(self.X) != len(self.y):
            raise ValueError('Length of data X and y must be the same')

        # Determine the number of features in the dataset
        self.n_features_in_ = self.X.shape[1]

        # Calculate hyperparameters based on the given values
        self.calculate_min_samples_leaf()
        self.calculate_min_samples_split()
        self.calculate_max_depth()
        self.calculate_max_features()

        # Determine the impurity criterion function
        self.calculate_criteria()

        if cost_matrix is not None:
            # Compute class counts
            class_counts = np.bincount(self.y)
            
            # Dummy initialization for leaf class counts
            leaf_class_counts = {0: class_counts}
            
            # Compute instance weights and class weight ratios based on the cost matrix
            self.instance_weights, self.class_weight_ratios = self.compute_instance_weights_and_ratios(
                cost_matrix, class_counts, leaf_class_counts
            )
        else:
            # If no cost matrix is provided, use default instance weights
            self.instance_weights = np.ones(len(self.y))
            self.class_weight_ratios = {}

        self.tree_ = self.build_tree(self.X, self.y, depth=0)

    def calculate_min_samples_leaf(self):
        """
        Determine the minimum number of samples required to be at a leaf node.
        """
        n_samples = self.X.shape[0]

        if isinstance(self.min_samples_leaf, float):
            # Ensure the float value is between 0 and 1
            if not (0 < self.min_samples_leaf < 1):
                raise ValueError("min_samples_leaf float type must be between 0 and 1")
            
            # Convert float to integer number of samples
            self.min_samples_leaf = int(np.ceil(self.min_samples_leaf * n_samples))

        elif isinstance(self.min_samples_leaf, int):
            # Ensure the integer value is greater than 0
            if self.min_samples_leaf <= 0:
                raise ValueError('min_samples_leaf must be greater than 0')
            
        else:
            raise ValueError("min_samples_leaf must be an int or a float")

    def build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters:
        -----------
        X : np.ndarray
            The training data.

        y : np.ndarray
            The target values.

        depth : int
            The current depth of the tree.

        Returns:
        --------
        tree : dict
            The constructed decision tree.
        """
        # Check stopping criteria
        if len(np.unique(y)) == 1:
            return {"class_counts": np.bincount(y)}

        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return {"class_counts": np.bincount(y)}

        # Determine the best split
        feature_index, threshold = self.calculate_splitter()(X, y)

        if feature_index is None:
            return {"class_counts": np.bincount(y)}

        # Split the data
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        left_tree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self.build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            "feature_index": feature_index,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree
        }

    def calculate_min_samples_split(self):
        """Determine the minimum number of samples required to split an internal node."""
        # all rows for sample
        n_samples = self.X.shape[0]

        # if float
        if isinstance(self.min_samples_split, float):
            # Ensure the float value is between 0 and 1
            if not (0 < self.min_samples_split < 1):
                raise ValueError("min_samples_split float type must be between 0 and 1")
            
            # Convert float to integer number of samples
            self.min_samples_split = int(np.ceil(self.min_samples_split * n_samples))
        
        # if int
        elif isinstance(self.min_samples_split, int):
            # Ensure the integer value is greater than 0
            if self.min_samples_split <= 0:
                raise ValueError('min_samples_split must be greater than 0')
        
        else:
            raise ValueError("min_samples_split must be an int or a float")

    def calculate_max_depth(self):
        """Validate the max_depth parameter"""
        # if int
        if isinstance(self.max_depth, int):
            # Ensure max_depth is greater than 0
            if self.max_depth <= 0:
                raise ValueError('max_depth must be greater than 0')
        
        elif self.max_depth is not None:
            raise ValueError('max_depth must be an int or None')

    def calculate_max_features(self):
        """
        Determine the number of features to consider for the best split.
        """
        if isinstance(self.max_features, int):
            # Ensure max_features is greater than 0
            if self.max_features <= 0:
                raise ValueError("max_features must be greater than 0")
            
            # Ensure max_features does not exceed the number of available features
            self.max_features = min(self.max_features, self.n_features_in_)
        
        elif isinstance(self.max_features, float):
            # Ensure the float value is between 0 and 1
            if not (0 < self.max_features < 1):
                raise ValueError("max_features float type must be between 0 and 1")
            
            # Convert float to integer number of features
            self.max_features = max(1, int(np.ceil(self.max_features * self.n_features_in_)))
        
        elif self.max_features == 'sqrt':
            # Use square root of the number of features
            self.max_features = int(np.sqrt(self.n_features_in_))
        
        elif self.max_features == 'log2':
            # Use base-2 logarithm of the number of features
            self.max_features = int(np.log2(self.n_features_in_))
        
        elif self.max_features is None:
            # Use all features
            self.max_features = self.n_features_in_
        
        else:
            raise ValueError("max_features must be an int, float, 'sqrt', 'log2' or None")

    def calculate_criteria(self):
        """
        Determine the function to measure the quality of a split.
        """
        if self.criterion == 'gini':
            self.criterion_function = self.gini_impurity

        elif self.criterion == 'log_loss':
            self.criterion_function = self.log_loss_impurity
        else:
            raise ValueError("Criterion not recognized. Use 'gini' or 'log_loss'.")

    def compute_instance_weights_and_ratios(self, cost_matrix, class_counts, leaf_class_counts):
        """
        Compute instance weights and class weight ratios based on the cost matrix.

        Parameters:
        -----------
        cost_matrix : np.ndarray
            A matrix where C[i, j] represents the cost of misclassifying an instance of class i as class j.

        class_counts : np.ndarray
            Count of instances for each class.

        leaf_class_counts : dict
            Dictionary of class counts in each leaf.

        Returns:
        --------
        instance_weights : np.ndarray
            The weight for each instance based on the cost matrix.

        class_weight_ratios : dict
            Dictionary of class weight ratios for each class.
        """
        # Initialize weights and ratios
        instance_weights = np.ones(len(self.y))
        class_weight_ratios = {}

        # Compute class weight ratios
        for i in range(len(cost_matrix)):

            total_cost = cost_matrix[i].sum()

            if total_cost > 0:

                class_weight_ratios[i] = cost_matrix[i] / total_cost

            else:
                class_weight_ratios[i] = np.zeros_like(cost_matrix[i])
        
        # Compute instance weights
        for idx, label in enumerate(self.y):
            class_weights = class_weight_ratios.get(label, np.zeros(len(cost_matrix)))
            instance_weights[idx] = class_weights.sum()

        return instance_weights, class_weight_ratios

    def gini_impurity(self, y):
        """
        Compute the Gini impurity of a set of labels.

        Parameters:
        -----------
        y : np.ndarray
            The target values.

        Returns:
        --------
        gini : float
            The Gini impurity of the labels.
        """
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def log_loss_impurity(self, y):
        """
        Compute the Logarithmic Loss (Log Loss) of a set of labels.

        Parameters:
        -----------
        y : np.ndarray
            The target values.

        Returns:
        --------
        log_loss : float
            The Logarithmic Loss of the labels.
        """
        proportions = np.bincount(y) / len(y)
        return -np.sum(proportions * np.log(proportions + 1e-10))

    def calculate_splitter(self):
        """
        Determine the strategy to find the best split.
        """
        if self.splitter == 'best':
            return self.best_split
        
        elif self.splitter == 'random':
            return self.random_split
        else:
            raise ValueError("Splitter not recognized. Use 'best' or 'random'.")

    def best_split(self, X, y):
        """
        Find the best split for the data based on the criterion function.

        Parameters:
        -----------
        X : np.ndarray
            The training data matrix where each row represents a sample and each column represents a feature.
        
        y : np.ndarray
            The target values corresponding to each sample in X.

        Returns:
        --------
        best_feature : int
            The index of the feature that results in the best split.

        best_threshold : float
            The threshold value for the best split.
        """
        best_score = float('inf')
        best_feature = None
        best_threshold = None

        # Iterate over all features to find the best split
        for feature_index in range(X.shape[1]):

            # Find unique threshold values for the current feature
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:

                # Create masks for the left and right splits
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                # Ensure both splits are non-empty
                if np.any(left_mask) and np.any(right_mask):

                    # Compute the impurity for each split using the criterion function
                    left_impurity = self.criterion_function(y[left_mask])
                    right_impurity = self.criterion_function(y[right_mask])
                    
                    # Compute the weighted average impurity for the split
                    score = (left_impurity * len(y[left_mask]) + right_impurity * len(y[right_mask])) / len(y)

                    # Update the best split if the current score is better
                    if score < best_score:
                        best_score = score
                        best_feature = feature_index
                        best_threshold = threshold

        return best_feature, best_threshold

    def random_split(self, X, y):
        """
        Choose a random split for the data.

        Parameters:
        -----------
        X : np.ndarray
            The training data matrix where each row represents a sample and each column represents a feature.
        
        y : np.ndarray
            The target values corresponding to each sample in X.

        Returns:
        --------
        random_feature : int
            The index of the randomly chosen feature for the split.

        random_threshold : float
            The threshold value for the random split.
        """
        # Randomly select a feature index
        feature_index = np.random.randint(0, X.shape[1])
        
        # Get unique threshold values for the selected feature
        thresholds = np.unique(X[:, feature_index])
        
        # Randomly select a threshold value
        threshold = np.random.choice(thresholds)

        return feature_index, threshold

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        -----------
        X : np.ndarray
            The input data.

        Returns:
        --------
        y_pred : np.ndarray
            The predicted class labels.
        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        return np.array([self._predict_sample(sample, self.tree_) for sample in X])

    def _predict_sample(self, sample, tree):
        """
        Predict the class label for a single sample.

        Parameters:
        -----------
        sample : np.ndarray
            The input sample.

        tree : dict
            The decision tree.

        Returns:
        --------
        class_label : int
            The predicted class label.
        """
        if 'class_counts' in tree:
            return np.argmax(tree['class_counts'])

        if sample[tree['feature_index']] <= tree['threshold']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])
    
    def predict_proba(self, X):
        """
        Predict the class probabilities for the given data.

        Parameters:
        -----------
        X : np.ndarray
            The input data.

        Returns:
        --------
        proba : np.ndarray
            The predicted class probabilities.
        """
        X = X.values if isinstance(X, pd.DataFrame) else X
        probas = np.zeros((X.shape[0], len(np.unique(self.y))))
        
        for i, sample in enumerate(X):
            class_counts = self._predict_sample_proba(sample, self.tree_)
            total = np.sum(class_counts)
            probas[i] = class_counts / total
        
        return probas

    def _predict_sample_proba(self, sample, tree):
        """
        Predict the class probabilities for a single sample.

        Parameters:
        -----------
        sample : np.ndarray
            The input sample.

        tree : dict
            The decision tree.

        Returns:
        --------
        class_counts : np.ndarray
            The class counts for the sample.
        """
        if 'class_counts' in tree:
            return tree['class_counts']

        if sample[tree['feature_index']] <= tree['threshold']:
            return self._predict_sample_proba(sample, tree['left'])
        else:
            return self._predict_sample_proba(sample, tree['right'])