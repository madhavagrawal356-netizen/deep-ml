import numpy as np

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing

    def forward(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes = np.unique(y)
        n_samples, n_features = X.shape

        # Log prior probabilities
        self.log_prior = {}

        # Log likelihoods P(X_j = 1 | class)
        self.log_prob = {}
        self.log_prob_neg = {}

        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)

            # Class prior
            self.log_prior[c] = np.log(n_c / n_samples)

            # Laplace-smoothed probability of feature being 1
            p = (np.sum(X_c, axis=0) + self.smoothing) / (
                n_c + 2 * self.smoothing
            )

            self.log_prob[c] = np.log(p)
            self.log_prob_neg[c] = np.log(1 - p)

    def predict(self, X):
        X = np.asarray(X)

        scores = []

        for c in self.classes:
            # Bernoulli log likelihood:
            # x*log(p) + (1-x)*log(1-p)
            likelihood = (
                X @ self.log_prob[c]
                + (1 - X) @ self.log_prob_neg[c]
            )

            score = self.log_prior[c] + likelihood
            scores.append(score)

        scores = np.array(scores).T

        # Select class with maximum posterior probability
        return self.classes[np.argmax(scores, axis=1)]