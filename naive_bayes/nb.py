import numpy as np


class NaiveBayes():

    def __init__(self):
        pass

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_class = len(self._classes)
        self._mean = np.zeros(shape=(n_class, n_features))
        self._variance = np.zeros(shape=(n_class, n_features))
        self._prior = np.zeros(shape=(n_class))

        for idx,c in enumerate(self._classes):
            X_c = X[y==c]
            self._mean[idx,:] = X_c.mean(axis=0)
            self._variance[idx,:] = X_c.var(axis=0)
            self._prior[idx] = X_c.shape[0] / n_samples

    def predict(self, X):
        ret = [self._predict(x) for x in X]
        return ret
    
    def _predict(self, x):
        posteriors = []
        for idx in range(len(self._classes)):
            log_prior = np.log(self._prior[idx]+0.00000001)
            class_conitional = np.sum(self._log_pdf(idx, x))
            posterior = log_prior + class_conitional
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _log_pdf(self, c_idx, x):
        mu= self._mean[c_idx]
        sigma = self._variance[c_idx] 
        ret = ((-0.5)*((x-mu)/sigma)**2) - np.log(sigma+0.00000001)
        return ret  


if __name__== "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import GaussianNB

    X,y = make_classification(n_samples=1000,n_features=10, n_classes=2, random_state=102)
    X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=0.33, random_state=102)

    #sklearn model
    nb_1 = GaussianNB()
    nb_1.fit(X_train,y_train)
    y_pred_1 = nb_1.predict(X_test)
    print("sklearn model:", accuracy_score(y_test,y_pred_1))

    # my model
    nb_2 = NaiveBayes()
    nb_2.fit(X_train,y_train)
    y_pred_2 = nb_2.predict(X_test)
    print("my model:", accuracy_score(y_test,y_pred_2))

