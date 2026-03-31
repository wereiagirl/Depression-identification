from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


def evaluate_model(clf, X_test, y_test, label_encoder):
    """
    只做评估，不涉及训练
    """

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========================")
    print("   模型评估结果")
    print("========================")

    print(f"\n[Accuracy] {acc:.2%}")

    print("\n[Classification Report]")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    print("\n[Confusion Matrix]")
    print(cm)

    return acc, cm