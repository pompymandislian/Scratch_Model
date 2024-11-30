import numpy as np

class ClassificationCriterion:
    @staticmethod
    def gini(y):
        """
        Compute the Gini impurity for a set of labels.

        Gini Impurity formula:
        Gini = 1 - sum(p_i^2)
        Where:
        p_i = proportion of samples belonging to class i.

        Parameters:
        y : array-like, shape (n_samples,)
            The labels of the samples.

        Returns:
        float
            The Gini impurity of the labels.
        """
        # Compute the probability of each class in the label
        unique, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)

        # Gini impurity formula: 1 - sum(p_i^2)
        return 1 - np.sum(prob ** 2)

    @staticmethod
    def log_loss(y):
        """
        Compute the Log Loss (cross-entropy) for a set of labels.

        Log Loss (Cross-Entropy) formula:
        Log Loss = - (1/N) * sum(y * log(p) + (1 - y) * log(1 - p))
        Where:
        p = predicted probability for class 1
        y = true label (0 or 1)
        N = number of samples

        Parameters:
        y : array-like, shape (n_samples,)
            The labels of the samples (binary classification).

        Returns:
        float
            The Log Loss (cross-entropy) of the labels.
        """
        # Assuming binary classification (labels 0 and 1)
        if len(np.unique(y)) == 1:
            return 0.0  # No uncertainty if all labels are the same

        # Using class probabilities as a proxy for prediction probabilities
        p = np.mean(y)  # Mean value as a simple prediction probability for class 1

        # Log Loss formula
        log_loss_value = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))

        return log_loss_value

    @staticmethod
    def entropy(y):
        """
        Compute the Entropy impurity for a set of labels.

        Entropy formula:
        Entropy = - sum(p_i * log2(p_i))
        Where:
        p_i = proportion of samples belonging to class i.

        Parameters:
        y : array-like, shape (n_samples,)
            The labels of the samples.

        Returns:
        float
            The Entropy impurity of the labels.
        """
        # Compute the probability of each class in the label
        unique, counts = np.unique(y, return_counts=True)
        prob = counts / len(y)

        # Entropy formula: - sum(p_i * log2(p_i))
        return -np.sum(prob * np.log2(prob + 1e-15))
