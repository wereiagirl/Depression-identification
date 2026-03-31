from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

from config import MODEL_DIR, TEST_SIZE
from data_loader import load_data
from preprocessing import create_wide_table
from feature_engineering import build_features
from model import build_model

def train_pipeline():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 加载数据
    df = load_data()

    # 2. 转宽表
    df_wide = create_wide_table(df)

    # 3. 特征工程
    X, y, scaler = build_features(df_wide)

    # 4. 切分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

    # 5. 模型
    model = build_model()
    model.fit(X_train, y_train)

    # 6. 评估
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"模型准确率: {acc:.4f}")

    # 7. 保存模型
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    return acc