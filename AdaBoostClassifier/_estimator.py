import numpy as np

class DecisionStump:
    def __init__(
        self, 
        feature_index=None, 
        threshold=None, 
        polarity=1
      ):
          self.feature_index = feature_index
          self.threshold = threshold
          self.polarity = polarity

    def fit(self, X, y, sample_weights):
        """
        Train the Decision Stump.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.
        sample_weights : array-like, shape (n_samples,)
            Sample weights.

        Returns:
        self : object
            Fitted DecisionStump classifier.
        """
        X = np.array(X).copy()
        y = np.array(y).copy()
        self.n_samples, self.n_features = X.shape

        # Best split result based on sample weights
        best_split_result = self._best_split_result(X, y, sample_weights)
        self.feature_index = best_split_result[0]
        self.threshold = best_split_result[1]
        self.polarity = best_split_result[2]

        return self

    def predict(self, X):
        """
        Predict the labels for the input data.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Data to predict.

        Returns:
        array-like, shape (n_samples,)
            Predicted labels.
        """
        X = np.array(X).copy()
        n_samples = X.shape[0]

        # Initialize predictions
        y_pred = np.ones(n_samples)

        # Apply polarity and threshold to make predictions
        if self.polarity == 1:
            y_pred[X[:, self.feature_index] <= self.threshold] = -1
        else:
            y_pred[X[:, self.feature_index] >= self.threshold] = 1

        return y_pred

    def _best_split_result(self, X, y, sample_weights):
        best_feature = None
        best_threshold = None
        best_polarity = None
        min_error = float('inf')

        # Searching for the best split
        for feature_i in range(self.n_features):

            # Find unique thresholds per column
            thresholds = np.unique(X[:, feature_i])

            for threshold in thresholds:
                for polarity in [1, -1]:

                    # Initialize predictions based on polarity
                    predictions = np.ones(self.n_samples)
                    if polarity == 1:
                        predictions[X[:, feature_i] <= threshold] = -1
                    else:
                        predictions[X[:, feature_i] >= threshold] = 1

                    # Calculate the weighted error
                    weighted_error = np.sum(sample_weights * (y != predictions))

                    # Correct the prediction polarity if the error is greater than 0.5
                    if weighted_error > 0.5:
                        polarity = -1
                        weighted_error = 1 - weighted_error

                    # Update the best split result
                    if weighted_error < min_error:
                        best_polarity = polarity
                        best_threshold = threshold
                        best_feature = feature_i
                        min_error = weighted_error

        return best_feature, best_threshold, best_polarity