import sys
from pathlib import Path

# 自动把 src 加入路径 → 修复 ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataset import dataset
from svm import train_model
from evaluate import evaluate

if __name__ == "__main__":
    csv_path = "./FINAL.csv"

    # 1. 数据准备
    X_train, X_test, y_train, y_test, le = dataset(csv_path)

    # 2. 训练
    clf = train_model(X_train, y_train)

    # 3. 评估
    evaluate(clf, X_test, y_test)

    # 特征重要性
    print("\nFeature Importances:")
    if hasattr(clf, "feature_importances_"):
        print(clf.feature_importances_)
    else:
        print("This model does not support feature importance.")