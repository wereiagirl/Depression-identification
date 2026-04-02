# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def prepare_data(csv_path):

    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig").dropna()
    except:
        df = pd.read_csv(csv_path, encoding="gbk").dropna()
    # Define columns
    ID_COL = "Subject_ID"
    LABEL_COL = "label"

    # Split subjects to avoid data leakage
    subjects = df[ID_COL].unique()
    train_subjects, test_subjects = train_test_split(
        subjects, test_size=0.2, random_state=42
    )

    train_df = df[df[ID_COL].isin(train_subjects)]
    test_df = df[df[ID_COL].isin(test_subjects)]

    # Features (remove ID and label)
    X_train = train_df.drop(columns=[ID_COL, LABEL_COL])
    X_test = test_df.drop(columns=[ID_COL, LABEL_COL])

    # Label encoding
    le = LabelEncoder()
    le.fit(df[LABEL_COL])

    y_train = le.transform(train_df[LABEL_COL])
    y_test = le.transform(test_df[LABEL_COL])

    print(f"[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    return X_train, X_test, y_train, y_test, le