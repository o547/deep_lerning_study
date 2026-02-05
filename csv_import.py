import pandas as pd
import numpy as np
df = pd.read_csv('./double.csv',  header=0)

# 複数列読む場合
# print(df.loc[:, 'original':'double'])

all_datas=df.loc[:, 'original':'original'].to_numpy().astype(np.float32)
all_labels=df.loc[:, 'double'].to_numpy().astype(np.float32)

datas=all_datas[::2]
labels=all_labels[::2]

test_datas=all_datas[1::2]
test_labels=all_labels[1::2]

