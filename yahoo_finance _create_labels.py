# モジュールのインポート
import pandas as pd
import numpy as np
import csv


# csvからimport
df = pd.read_csv('./ticker5y.csv',  header=0)

all_sp=df.iloc[0:, 1:].to_numpy().astype(np.float32).T

all_datas = []

for sp in all_sp:
    for i in range(1,len(sp)-66):
        all_datas.append(sp[i:i+65].tolist())

with open('65days.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(all_datas)