import pandas as pd

# 文件路径
path_sp = r"d:\my_python_project\All_Subjects_SP_Features.csv"
path_scale = r"d:\my_python_project\329.csv"

# 读取 SP 特征表
df_sp = pd.read_csv(path_sp, encoding="utf-8-sig")

# 读取量表（解决编码报错）
df_scale = pd.read_csv(path_scale, encoding="gbk")
df_scale = df_scale[["皮肤电势编号", "抑郁量表总分"]]

# 合并数据
df_final = pd.merge(
    df_sp,
    df_scale,
    left_on="Subject_ID",
    right_on="皮肤电势编号",
    how="left"
)

# 删除多余列
df_final = df_final.drop(columns=["皮肤电势编号"])

# 保存最终文件
df_final.to_csv(
    r"d:\my_python_project\FINAL.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Done")