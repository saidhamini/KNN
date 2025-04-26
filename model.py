from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_model(X_train, X_test, y_train, y_test, hyperparams):
    try:
        model = KNeighborsClassifier(**hyperparams)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return {"accuracy": acc}
    except Exception as e:
        return {"error": str(e)}
