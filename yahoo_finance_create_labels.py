# モジュールのインポート
import pandas as pd
import numpy as np
import csv
import random
import scipy.stats


# csvからimport
df = pd.read_csv("./ticker5y.csv", header=0)

all_sp = df.iloc[0:, 1:].to_numpy().astype(np.float32).T

all_datas = []

for sp in all_sp:
    if random.random() < 0.6:
        continue
    for i in range(1, len(sp) - 67):
        column = scipy.stats.zscore(sp[i : i + 65].tolist()).tolist()
        if sp[i + 66] > sp[i + 65]:
            column.append(0)
        else:
            column.append(1)

        all_datas.append(column)

with open("65days.csv", "w") as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerows(all_datas)
