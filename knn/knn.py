import numpy as np 
from collections import Counter
from utils import eculidean_distance

class KNN():
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self,X):
        labels = [self._predict_one_sample_from_all(x) for x in X]
        return labels

    def _predict_one_sample_from_all(self, x):
        distances = [eculidean_distance(x, x_train) for x_train in self.X_train] 
        idx = np.argpartition(np.array(distances), self.n_neighbors)
        class_label = [self.y_train[i] for i in  idx[:self.n_neighbors]]
        majority = Counter(class_label).most_common()[0][0]
        return majority 

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    X,y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                        test_size=0.2,
                        random_state=102)

    #sklearn model
    knn_1 = KNeighborsClassifier(n_neighbors=5)
    knn_1.fit(X_train,y_train)
    y_pred_1 = knn_1.predict(X_test)
    print("sklearn_knn: ",accuracy_score(y_test,y_pred_1))

    #my knn
    knn_2 = KNN(n_neighbors=5)
    knn_2.fit(X_train,y_train)
    y_pred_2 = knn_2.predict(X_test)
    print("my_knn: ",accuracy_score(y_test,y_pred_2))

