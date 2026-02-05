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



# カスタムデータセットの作成
class CustomDataset(Dataset):
    def __init__(self, datas, labels):
        # データとラベルを受け取るコンストラクタ
        self.datas = datas
        self.labels = labels

    def __len__(self):
        # データセットのサイズを返す
        return len(self.datas)

    def __getitem__(self, idx):
        # 指定したインデックスのデータとラベルを返す
        sample = {'datas': self.datas[idx], 'label': self.labels[idx]}
        return sample

# ダミーデータを作成

df = pd.read_csv('./double.csv',  header=0)
all_datas=df.loc[:, 'original':'original'].to_numpy().astype(np.float32)
all_labels=df.loc[:, 'double'].to_numpy().astype(np.float32)


datas = all_datas[::2]
labels = all_labels[::2]

test_datas = all_datas[::2]
test_labels = all_labels[::2]


# データセットのインスタンスを作成
custom_dataset = CustomDataset(datas, labels)
test_dataset = CustomDataset(test_datas, test_labels)


# DataLoaderの作成
data_loader = DataLoader(dataset=custom_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)


class mlp_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc15 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)


        x = self.fc15(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x
    

def train(model, train_loader):
    # 今は学習時であることを明示するコード
    model.train()
    for i in range(30):
        # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
        for batch in train_loader:
            batch_imgs=batch['datas']
            batch_labels=batch['label']
            batch_imgs = batch_imgs.reshape(10, 1)
            batch_labels = batch_labels.reshape(10, 1)

            outputs = model(batch_imgs)  # 順伝播
            model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
            # print(outputs)
            # print(batch_labels)
            loss = model.criterion(outputs, batch_labels)  # 損失を計算
            loss.backward()  # 逆伝播で勾配を計算
            model.optimizer.step()  # 最適化
        if(i%10==0):
            print(str(i)+"周目終了")
       



def test(model, train_loader):
    # 今は学習時であることを明示するコード
    model.eval()

    # 正しい予測数、損失の合計、全体のデータ数を数えるカウンターの0初期化
    total_correct = 0
    total_loss = 0
    total_data_len = 0

    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
    for batch in train_loader:
        batch_imgs=batch['datas']
        batch_labels=batch['label']
        batch_imgs = batch_imgs.reshape(10, 1)
        batch_labels = batch_labels.reshape(10, 1)
        outputs = model(batch_imgs)  # 順伝播
        loss = model.criterion(outputs, batch_labels)  # 損失を計算
       
        batch_size = len(batch_labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            if(total_data_len%100==0):
                #print("datas:"+str(batch_imgs[i].item())+"  out:"+str(outputs[i].item())+"  label:"+str(batch_labels[i].item())+"  difference:"+str((outputs[i].item() - batch_labels[i].item())**2))
                None
            total_data_len += 1  # 全データ数を集計
            if (outputs[i].item() - batch_labels[i].item())**2 < 100:
                total_correct += 1 # 正解のデータ数を集計
        total_loss += loss.item()  # 全損失の合計

    # 今回のエポックの正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    loss = total_loss/total_data_len  # 損失の平均の算出
    return accuracy, loss


# モデルを宣言する
model = mlp_net()

# 学習させ、その結果を表示する
train(model, data_loader)

acc, loss = test(model, test_loader)
print(f'正答率: {acc}, 損失: {loss}')