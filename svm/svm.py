import numpy as np

class SVM():

    def __init__(self,_iter=1000, lr=0.001, alpha=0.1):

        self._iter = _iter
        self.lr = lr
        self.alpha = alpha
        self.w = None
        self.b = None

    def fit(self, X, y):
        _, n_features = X.shape
        y_ = np.where(y<=0, -1, 1)

        self.w = np.zeros(shape=(n_features))
        self.b = 0
        for _ in range(self._iter):
            for idx, x in enumerate(X):
                condtion = y_[idx] * self._f(x) >= 1
                if condtion:
                    delta_w = 2* self.alpha * self.w
                    # b does not change
                    self.w = self.w - self.lr * delta_w 
                else:
                    delta_w = 2* self.alpha * self.w - y_[idx] * x
                    delta_b = y_[idx] 
                    self.w = self.w - self.lr * delta_w 
                    self.b = self.b - self.lr * delta_b

    def predict(self, X):
        ret = np.dot(X, self.w) - self.b
        return np.sign(ret)


    def _f(self,x):
        return np.dot(self.w, x) - self.b

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import SGDClassifier

    X, y = make_blobs(n_samples=100,n_features=2, random_state=102,centers=2,   
        cluster_std=1.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                        test_size=0.2,
                        random_state=102)

    train_test_split
    #sklearn model
    cls_1 = SGDClassifier(alpha=0.01)
    cls_1.fit(X_train,y_train)
    y_pred_1 = cls_1.predict(X_test)
    print("sklearn_SGD_SVM: ",accuracy_score(y_test,y_pred_1))

    cls_2 = SGDClassifier(alpha=0.01)
    cls_2.fit(X_train,y_train)
    y_pred_2 = cls_2.predict(X_test)
    print("MY_SGD_SVM: ",accuracy_score(y_test,y_pred_2))