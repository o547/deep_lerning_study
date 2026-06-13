import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 読み込み
df = pd.read_csv("./ticker5y.csv", header=0)

# 標準化
scaler = StandardScaler()
df = scaler.fit_transform(df)

# 出力
df.to_csv("Dengue_diseases_dataset_pretreatment.csv", header=True, index=False)
