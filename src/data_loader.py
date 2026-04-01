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
            # 先尝试 UTF-8 读取行
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # 读取表格（无错误参数）
            df = pd.read_csv(
                file_path,
                skiprows=9,
                header=None,
                encoding='utf-8',
                on_bad_lines='skip'
            )
            df = df.iloc[:, :3]
            df.columns = ['Date', 'Time', 'SP_Value']

            dataset[subject_id] = {
                "lines": lines,
                "dataframe": df
            }
            print(f"[LOADED] {subject_id}")

        except:
            try:
                # 兜底 GBK
                with open(file_path, 'r', encoding='gbk', errors='ignore') as f:
                    lines = f.readlines()

                df = pd.read_csv(
                    file_path,
                    skiprows=9,
                    header=None,
                    encoding='gbk',
                    on_bad_lines='skip'
                )
                df = df.iloc[:, :3]
                df.columns = ['Date', 'Time', 'SP_Value']

                dataset[subject_id] = {
                    "lines": lines,
                    "dataframe": df
                }
                print(f"[LOADED] {subject_id} (gbk fallback)")

            except Exception as e:
                print(f"[FAILED] {subject_id}: {str(e)}")

    return dataset