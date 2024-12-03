import numpy as np

class AdaboostClassifier:
    """
    AdaBoost classifier for binary classification.

    Parameters:
    estimator : object, optional
        Base estimator to use, by default None which uses DecisionStump.
    n_estimators : int, optional
        The number of estimators to train, by default 5.
    weighted_sampled : bool, optional
        If True, sample weights are updated, by default False.
    """

    def __init__(
        self,
        estimator=None,
        n_estimators=5,
        weighted_sampled=False # focus weight
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.weighted_sampled = weighted_sampled
        self.models = []  # Initialize the models list to store the models

    def fit(self, X, y):
        """
        Train the AdaBoost classifier by fitting the base estimator.

        Parameters:
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target labels.

        Returns:
        self : object
            Fitted AdaBoost classifier.
        """
        if self.estimator is None:
            base_estimator = DecisionStump()

        # Convert to array
        X = np.array(X).copy()
        y = np.array(y).copy()

        self.n_samples, self.n_features = X.shape

        # All weights are the same initially
        self.sample_weights = np.ones(self.n_samples) / self.n_samples
        self.alphas = np.zeros(self.n_estimators)

        # Training loop
        for i in range(self.n_estimators):
            model = base_estimator.fit(X, y, sample_weights=self.sample_weights)
            y_pred = model.predict(X)
            error = self._weighted_error(self.sample_weights, y, y_pred)

            # Calculate alpha
            alpha = 0.5 * np.log((1 - error) / max(error, 1e-10))  # Avoid division by 0
            self.alphas[i] = alpha

            # Update sample weights
            if self.weighted_sampled:
                self.sample_weights = self.sample_weights * np.exp(-alpha * y * y_pred)
                self.sample_weights /= np.sum(self.sample_weights)  # Normalize weights

            self.models.append(model)

        return self

    def _weighted_error(self, weights, y, y_pred):
        """
        Calculate the weighted error of the predictions.

        Parameters:
        weights : array-like, shape (n_samples,)
            Sample weights.
        y : array-like, shape (n_samples,)
            True labels.
        y_pred : array-like, shape (n_samples,)
            Predicted labels.

        Returns:
        float
            Weighted error rate.
        """
        return np.sum(weights * (y != y_pred)) / np.sum(weights)

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
        y_pred = np.zeros(n_samples)

        # Sum predictions from all estimators
        for i in range(self.n_estimators):
            y_pred += self.alphas[i] * self.models[i].predict(X)

        return np.sign(y_pred)