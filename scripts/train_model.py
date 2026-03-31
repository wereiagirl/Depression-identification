import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import prepare_data, train_model
from evaluate import evaluate_model


if __name__ == "__main__":

    csv_path = "./outputs/All_Subjects_SP_Features.csv"

    # 1. 数据准备
    X_train, X_test, y_train, y_test, le = prepare_data(csv_path)

    # 2. 训练
    clf = train_model(X_train, y_train)

    # 3. 评估
    evaluate_model(clf, X_test, y_test, le)