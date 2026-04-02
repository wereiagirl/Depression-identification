import os
import pandas as pd


def reshape_features(input_file, output_dir="./outputs"):

    if not os.path.exists(input_file):
        print(f"[ERROR] File not found: {input_file}")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_file)
    print(f"[INFO] Loaded feature table: {len(df)} rows")

    print("\n--- Generating Stage-wise tables ---")

    stages = df['Stage'].unique()

    for stage in stages:
        stage_df = df[df['Stage'] == stage].copy()

        if 'Stage' in stage_df.columns:
            stage_df.drop(columns=['Stage'], inplace=True)

        out_path = os.path.join(output_dir, f"Features_Task_{stage}.csv")
        stage_df.to_csv(out_path, index=False, encoding='utf-8-sig')

        print(f"[OK] {out_path} ({len(stage_df)} rows)")

    print("\n--- Generating Wide Format Table ---")

    df_wide = df.pivot(index='Subject_ID', columns='Stage')


    df_wide.columns = [
        f"{stage}_{feature}"
        for feature, stage in df_wide.columns
    ]

    df_wide.reset_index(inplace=True)

    wide_out = os.path.join(output_dir, "Features_Wide_All_Tasks.csv")
    df_wide.to_csv(wide_out, index=False, encoding='utf-8-sig')

    print(f"[OK] {wide_out}")
    print("\n[DONE] Feature reshaping completed.")


if __name__ == "__main__":
    input_file = "./outputs/All_Subjects_SP_Features.csv"
    reshape_features(input_file)