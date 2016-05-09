from sklearn.naive_bayes import BaseNB, check_X_y, check_array
import numpy as np


class PoissonNB(BaseNB):
    """
    Attributes
    --------
    class_prior_ : array, shape(n_class,)
        Probability of each class

    class_count_ : array, shape(n_class,)
        Number of training Sample in each class

    lambda_ : array, shape(n_class, n_features)
        mean of each feature per class

    """

    def fit(self, X, y):
        """
        :param X: Design Matrix, shape(n_trials,n_features)
        :param y: Response Vector, shape(n_trials,)
        :return: self
        """

        X, y = check_X_y(X, y)
        PoissonNB.check_non_negative(X)

        n_trials, n_features = X.shape

        self.classes_ = unique_y = np.unique(y)

        n_classes = unique_y.shape[0]

        epsilon = 1e-9

        self.lambda_ = np.zeros((n_classes, n_features))

        self.class_prior_ = np.zeros(n_classes)

        for i, y_i in enumerate(n_classes):
            Xi = X[y == y_i, :]
            self.lamda_[i, :] = np.mean(Xi, axis=0) + epsilon
            self.class_prior_[i] = float(Xi.shape[0])/n_trials

        return self

    def _joint_log_likelihood(self, X):
        X = check_array(X)
        PoissonNB.check_non_negative(X)

        joint_log_likelihood = np.zeros((np.shape(X)[0],len(self.classes_)))

        for i in range(len(self.classes_)):
            n_ij = np.sum(X*np.log(self.lambda_[i, :]), axis=1)
            n_ij -= np.sum(self.lambda_[i, :])
            joint_log_likelihood[:, i] = np.log(self.prior_[i]) + n_ij

        return joint_log_likelihood

    @staticmethod
    def check_non_negative(X):
        if np.any(X < 0.):
            raise ValueError("Input X must be non-negative")
