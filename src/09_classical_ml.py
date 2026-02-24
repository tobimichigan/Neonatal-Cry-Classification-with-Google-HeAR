from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm(train_X, train_y, val_X, val_y):
    model = SVC(kernel="rbf", probability=True)
    model.fit(train_X, train_y)
    val_acc = accuracy_score(val_y, model.predict(val_X))
    return model, val_acc
