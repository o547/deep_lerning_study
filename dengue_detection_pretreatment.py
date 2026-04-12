import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 読み込み
df = pd.read_csv("./Dengue_diseases_dataset_modified.csv", header=0)

# 平均値で欠損値処理
df = df.fillna(df.mean(numeric_only=True))

# 置換
# df = df.replace({"F": 0, "M": 1, "No": 0, "Yes": 1})
# one-hot-encoding
df = pd.get_dummies(df, columns=["gender"], dtype='uint8')

#標準化
num_cols = ["age","hemoglobin_g_dl","wbc_count","rbc_count","platelet_count","platelet_distribution_width"]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

#正解ラベルを最後の列にする
df = df[[c for c in df.columns if c != 'dengue_label'] + ['dengue_label']]

#出力
df.to_csv("Dengue_diseases_dataset_pretreatment.csv", header=True, index=False)
