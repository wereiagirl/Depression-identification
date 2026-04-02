import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv("outputs/Features_Wide_All_Tasks.csv")

print("Missing values BEFORE imputation:")
print(df.isna().sum().sum())

id_col = df["Subject_ID"]
label_col = df["label"] if "label" in df.columns else None

X = df.drop(columns=["Subject_ID"] + (["label"] if "label" in df.columns else []))

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# Convert back to DataFrame
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

df_filled = pd.concat([id_col, X_imputed], axis=1)

if label_col is not None:
    df_filled["label"] = label_col.values

df_filled.to_csv("outputs/Features_Wide_All_Tasks_filled.csv", index=False, encoding="utf-8-sig")

print("Missing values AFTER imputation:")
print(df_filled.isna().sum().sum())

print("Imputation completed and file saved!")