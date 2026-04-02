from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(clf, X_test, y_test, label_encoder):
    """
    只做评估，不涉及训练
    包含：准确率、AUC、分类报告、混淆矩阵、图示
    """

    # 预测
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    # 输出结果（完全仿照你的格式）
    print("\n========================")
    print("evaluate result")
    print("========================")

    print(f"\n[Accuracy] {acc:.2%}")
    print(f"[AUC] {auc:.4f}")

    print("\n[Classification Report]")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    print("\n[Confusion Matrix]")
    print(cm)

    # ====================== 图示：混淆矩阵热力图 ======================
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,        # 显示数字
        fmt="d",           # 整数格式
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.xlabel("Predict Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix | AUC={auc:.4f}")
    plt.tight_layout()

    # 自动保存图片（方便论文/报告）
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()  # 弹出图片

    return acc, auc, cm