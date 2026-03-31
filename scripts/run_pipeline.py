import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_raw_files
from preprocessing import segment_and_preprocess_sp
from feature_engineering import extract_features


if __name__ == "__main__":

    data_dir = "./data"

    print("Loading raw data...")
    raw_data = load_raw_files(data_dir)

    dataset = {}
    all_features = []

    print(f"Total subjects: {len(raw_data)}. Starting processing...")

    for subject_id, content in raw_data.items():

        try:
            # ===== preprocessing =====
            segments = segment_and_preprocess_sp(
                content["lines"],
                content["dataframe"]
            )

            dataset[subject_id] = segments

            # ===== feature extraction =====
            subject_features = extract_features(segments, subject_id)
            all_features.extend(subject_features)

            print(f"[SUCCESS] Processed: {subject_id}")

        except Exception as e:
            print(f"[ERROR] Skipped: {subject_id} | Reason: {str(e)}")

    # ===== save =====
    if len(all_features) > 0:
        final_df = pd.DataFrame(all_features)

        output_file = "./outputs/All_Subjects_SP_Features.csv"
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print("\n[DONE] Processing complete!")
        print(f"[INFO] Total feature rows: {len(final_df)}")
        print(f"[INFO] Saved to: {output_file}")

    else:
        print("\n[FAILED] No valid features extracted.")