import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class Validator:
    def __init__(self, classifier: BaseEstimator):
        self.classifier = classifier

    def evaluate(self, X: np.ndarray, y: np.ndarray, feature_subset: list) -> float:
        X_subset = X[:, feature_subset]
        accuracy = []
        for i in range(len(X_subset)):
            X_train = np.concatenate((X_subset[:i], X_subset[i+1:]))
            y_train = np.concatenate((y[:i], y[i+1:]))
            X_test, y_test = np.array([X_subset[i]]), np.array([y[i]])
            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
        return np.mean(accuracy)

# Trace using sample
X = np.array([
    [0.01, 0.02, 0.02],
    [0.01, 0.01, 0.03],
    [0.02, 0.03, 0.02],
    [0.03, 0.02, 0.02],
    [0.05, 0.01, 0.05]
])
y = np.array([1, 2, 1, 1, 2])

# Example classifier
classifier = KNeighborsClassifier(2)
validator = Validator(classifier)
accuracy = validator.evaluate(X, y, [0, 1, 2]) * 100

print(f"Accuracy: {accuracy}%")