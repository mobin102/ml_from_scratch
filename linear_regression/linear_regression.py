import numpy as np

class LinearRegression():

    def __init__(self, lr=0.009, iter_=1000, batch_size=None):
        self.lr = lr
        self.iter = iter_
        self.batch_size = batch_size

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.n_samples, self.n_features = self.X_train.shape
        self.weight, self.bais = self._gd()

    def predict(self, X):
        y_hat = np.dot(X, self.weight) + self.bais
        return y_hat
    def _gd(self):
        weight = np.random.normal(size=(self.n_features))
        bais = np.random.normal()
        y_hat = np.dot(self.X_train, weight) + bais
        for _ in range(self.iter):
            e = y_hat - self.y_train
            delta_w = (1/self.n_samples) * np.dot(self.X_train.T,e) 
            delta_b = (1/self.n_samples) * np.sum(e)

            weight = weight - self.lr * delta_w
            bais = bais - self.lr * delta_b
            y_hat = np.dot(self.X_train, weight) + bais

        return weight, bais


if __name__ == "__main__":
     from sklearn.datasets import make_regression
     from sklearn import linear_model
     from sklearn.model_selection import train_test_split
     from sklearn.metrics import mean_squared_error
#    import matplotlib.pyplot as plt
     X,y = make_regression(n_samples=200,n_features=1,noise=5)
     X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=102)      

     #sklearn model
     lreg_1 = linear_model.SGDRegressor(loss="squared_loss", alpha=0)
     lreg_1.fit(X_train,y_train)
     pred_1 = lreg_1.predict(X_test)
     print("sklearn model",round(mean_squared_error(y_test,pred_1),3))
     
     
# =============================================================================
#      plt.scatter(X,y)
#      plt.plot(X,pred,c="r")
#      plt.show()
# =============================================================================
     
     #mymodel
     lreg_2 = LinearRegression()
     lreg_2.fit(X_train,y_train)
     pred_2 = lreg_2.predict(X_test)
     print("my model",round(mean_squared_error(y_test,pred_2),3))
# =============================================================================
#      plt.scatter(X,y)
#      plt.plot(X,pred,c="g")
#      plt.show()
# =============================================================================
