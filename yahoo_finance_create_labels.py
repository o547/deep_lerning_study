# モジュールのインポート

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import utils
from torchvision import datasets
import torchvision.transforms as transforms

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv
import pprint


# csvからimport
df = pd.read_csv('./ticker5y.csv',  header=0)

all_sp=df.iloc[0:, 1:].to_numpy().astype(np.float32).T

all_datas = []

for sp in all_sp:
    for i in range(1,len(sp)-22):
        all_datas.append(sp[i:i+21].tolist())

with open('21days.csv', 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(all_datas)