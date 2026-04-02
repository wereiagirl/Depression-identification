import sys
from pathlib import Path

# 把 src 加入 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from dataset import dataset


if __name__ == "__main__":
    csv_path = "./FINAL.csv"

    # 1. 数据准备
    X_train, X_test, y_train, y_test, le = dataset(csv_path)

    # ===== 2. 原始特征 AUC =====
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
    auc_before = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
    print(f"[AUC BEFORE] {auc_before:.4f}")

    # ===== 3. 特征筛选 =====
    selector = SelectKBest(f_classif, k=7)
    X_train_new = selector.fit_transform(X_train, y_train)
    X_test_new = selector.transform(X_test)

    # ===== 4. 筛选后 AUC =====
    auc_after = cross_val_score(model, X_train_new, y_train, cv=5, scoring='roc_auc').mean()
    print(f"[AUC AFTER] {auc_after:.4f}")

    # ===== 5. 输出被选中的特征索引 =====
    selected_indices = selector.get_support(indices=True)
    print("\n[Selected Feature Indices]")
    print(selected_indices)

    # ===== 6. 输出每个特征的重要性评分 =====
    print("\n[Feature Scores (F-score)]")
    print(selector.scores_)