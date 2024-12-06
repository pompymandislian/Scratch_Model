import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

class EUSBoost(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=50, random_state=None):
        self.base_estimator = base_estimator if base_estimator is not None else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.random_state = random_state
    
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weights_dict = dict(zip(np.unique(y), class_weights))
        
        self.estimators_ = []
        self.alpha_ = []
        
        # Initialize weights
        sample_weights = np.ones(n_samples) / n_samples
        
        for t in range(self.n_estimators):
            # Fit the base estimator
            estimator = self.base_estimator
            estimator.fit(X, y, sample_weight=sample_weights)
            
            # Compute the predictions
            y_pred = estimator.predict(X)
            
            # Compute the error
            error = np.sum(sample_weights * (y_pred != y)) / np.sum(sample_weights)
            
            # Compute alpha
            alpha = np.log((1 - error) / (error + 1e-10))  # avoid division by zero
            self.estimators_.append(estimator)
            self.alpha_.append(alpha)
            
            # Update sample weights
            sample_weights *= np.exp(alpha * (y_pred != y))
            sample_weights /= np.sum(sample_weights)
        
        return self
    
    def predict(self, X):
        # Aggregate the predictions from the base estimators
        pred_sum = np.zeros(X.shape[0])
        for estimator, alpha in zip(self.estimators_, self.alpha_):
            pred_sum += alpha * estimator.predict(X)
        
        return np.sign(pred_sum)
    
    def predict_proba(self, X):
        # Initialize probability sums
        prob_sum = np.zeros((X.shape[0], self.estimators_[0].predict_proba(X).shape[1]))
        for estimator, alpha in zip(self.estimators_, self.alpha_):
            prob_sum += alpha * estimator.predict_proba(X)
        
        # Normalize probabilities
        prob_sum = prob_sum / np.sum(self.alpha_)
        
        # Ensure probabilities sum to 1
        prob_sum = np.clip(prob_sum, 0, 1)
        row_sums = prob_sum.sum(axis=1, keepdims=True)
        prob_sum /= row_sums
        
        return prob_sum