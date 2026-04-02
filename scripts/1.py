import pandas as pd

# ======================
# 1. 文件路径
# ======================
path_sp = r"d:\my_python_project\Features_Wide_All_Tasks_filled.csv"
path_scale = r"d:\my_python_project\329.csv"

# ======================
# 2. 读取数据
# ======================
df_sp = pd.read_csv(path_sp, encoding="utf-8-sig")

df_scale = pd.read_csv(path_scale, encoding="gbk")
df_scale = df_scale[["皮肤电势编号", "抑郁量表总分"]]

# ======================
# 3. 合并
# ======================
df_final = pd.merge(
    df_sp,
    df_scale,
    left_on="Subject_ID",
    right_on="皮肤电势编号",
    how="left"
)

# ======================
# 4. 删除多余列
# ======================
df_final = df_final.drop(columns=["皮肤电势编号"])

# ======================
# 5. 生成二值标签（关键！）
# ======================
# 推荐阈值：>=10 判为抑郁

df_final["label"] = (df_final["抑郁量表总分"] >= 10).astype(int)

# ======================
# 6. 检查类别分布（很重要）
# ======================
print("Label distribution:")
print(df_final["label"].value_counts())

# ======================
# 7. 检查是否有未匹配成功的数据
# ======================
missing_label = df_final["抑郁量表总分"].isna().sum()
print(f"Missing scale scores: {missing_label}")

# 如果有缺失，可以选择删除
df_final = df_final.dropna(subset=["抑郁量表总分"])

# ======================
# 8. 保存最终宽表
# ======================
df_final.to_csv(
    r"d:\my_python_project\FINAL_binary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Binary dataset saved successfully!")