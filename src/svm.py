from sklearn.svm import SVC

def train_model(X_train, y_train):
    """
    Unified training interface for SVM
    """
    clf = SVC(
        kernel="rbf",          # 核函数
        probability=True,      # 必须开启，才能输出概率 → 支持 AUC
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_train, y_train)
    return clf