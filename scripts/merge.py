# -*- coding: utf-8 -*-
"""
Data Merging Script
Merge SP features + scale scores → generate FINAL_binary.csv
"""
import pandas as pd

def merge_data():

    path_sp = "./Features_Wide_All_Tasks_filled.csv"
    path_scale = "./329.csv"

    # Load data
    df_sp = pd.read_csv(path_sp, encoding="utf-8-sig")
    df_scale = pd.read_csv(path_scale, encoding="gbk")

    # Select only necessary columns
    df_scale = df_scale[["皮肤电势编号", "抑郁量表总分"]]

    # Merge
    df_final = pd.merge(
        df_sp,
        df_scale,
        left_on="Subject_ID",
        right_on="皮肤电势编号",
        how="left"
    )

    # Drop merge key
    df_final = df_final.drop(columns=["皮肤电势编号"])

    # Create binary label (>=10 → depression=1)
    df_final["label"] = (df_final["抑郁量表总分"] >= 4).astype(int)

    # Show label distribution
    print("Label distribution:")
    print(df_final["label"].value_counts())

    # Missing values
    missing_label = df_final["抑郁量表总分"].isna().sum()
    print(f"Missing scale scores: {missing_label}")

    df_final = df_final.drop(columns=["抑郁量表总分"])
    # Drop missing labels
    df_final = df_final.dropna(subset=["label"])

    # Save final dataset
    df_final.to_csv(
        "./FINAL.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("FINAL.csv saved successfully!")

if __name__ == "__main__":
    merge_data()