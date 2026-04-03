from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n========================")
    print("evaluate result")
    print("========================")

    print(f"\n[Accuracy] {acc:.2%}")
    print(f"[AUC] {auc:.4f}")

    target_names = ["negative", "positive"]

    print("\n[Classification Report]")
    print(classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    ))

    print("\n[Confusion Matrix]")
    print(cm)


    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix | AUC={auc:.4f}")
    plt.tight_layout()


    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.show()

    return acc, auc, cm