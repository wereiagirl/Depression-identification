import os
import glob
import pandas as pd
from typing import Dict


def load_raw_files(data_dir: str) -> Dict:

    file_paths = glob.glob(os.path.join(data_dir, "*.csv"))

    if not file_paths:
        raise ValueError(f"No CSV files found in {data_dir}")

    dataset = {}

    for file_path in file_paths:
        subject_id = os.path.splitext(os.path.basename(file_path))[0]

        try:
            # ===== read raw lines =====
            with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
                lines = f.readlines()

            # ===== read table =====
            df = pd.read_csv(file_path, skiprows=9, header=None, encoding='gb18030')
            df = df.iloc[:, :3]
            df.columns = ['Date', 'Time', 'SP_Value']

            dataset[subject_id] = {
                "lines": lines,
                "dataframe": df
            }

            print(f"[LOADED] {subject_id}")

        except Exception as e:
            print(f"[FAILED] {subject_id}: {e}")

    return dataset