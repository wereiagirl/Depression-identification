from sklearn.svm import SVC

def train_model(X_train, y_train):

    clf = SVC(
        kernel="rbf",
        C=1.4 ,
        gamma=0.0085,
        random_state=42,
        probability=True,
        class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    return clf