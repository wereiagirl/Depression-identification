import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

def dataset(csv_path):
    # 读取数据
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna()
    except:
        df = pd.read_csv(csv_path, encoding="gbk").dropna()

    ID_COL = "Subject_ID"
    LABEL_COL = "label"

    #
    subjects = df[ID_COL].unique()
    train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=42)

    train_df = df[df[ID_COL].isin(train_subjects)]
    test_df = df[df[ID_COL].isin(test_subjects)]

    #
    feature_cols = df.drop(columns=[ID_COL, LABEL_COL]).columns
    selected_idx = [6, 9, 10, 13, 14, 15, 16, 18, 19, 21] # 已确定的有效特征
    selected_features = feature_cols[selected_idx]

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]

    #
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[LABEL_COL])
    y_test = le.transform(test_df[LABEL_COL])

    #
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #
    return X_train, X_test, y_train, y_test, le