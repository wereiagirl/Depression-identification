import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


def prepare_data(csv_path):
    """
    数据读取 + 二分类筛选 + 按被试划分
    """

    df = pd.read_csv(csv_path).dropna()

    # 二分类：Positive vs Negative
    df = df[df['Stage'].isin(['Positive', 'Negative'])]

    # 防止数据泄漏（按人划分）
    subjects = df['Subject_ID'].unique()
    train_sub, test_sub = train_test_split(
        subjects, test_size=0.2, random_state=42
    )

    train_df = df[df['Subject_ID'].isin(train_sub)]
    test_df = df[df['Subject_ID'].isin(test_sub)]

    # 特征
    X_train = train_df.drop(columns=['Subject_ID', 'Stage'])
    X_test = test_df.drop(columns=['Subject_ID', 'Stage'])

    # 标签编码
    le = LabelEncoder()
    le.fit(df['Stage'])

    y_train = le.transform(train_df['Stage'])
    y_test = le.transform(test_df['Stage'])

    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    return X_train, X_test, y_train, y_test, le


def train_model(X_train, y_train):
    """
    只做训练，不做任何评估
    """

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X_train, y_train)

    return clf