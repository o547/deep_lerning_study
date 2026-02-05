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



# csvからimport
df = pd.read_csv('./21days.csv',  header=0)

all_datas=df.iloc[0:, 0:20].to_numpy().astype(np.float32)
all_labels=df.iloc[0:, 20].to_numpy().astype(np.float32)


datas = all_datas[::2]
labels = all_labels[::2]


test_datas = all_datas[1::2]
test_labels = all_labels[1::2]




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



# データセットのインスタンスを作成
custom_dataset = CustomDataset(datas, labels)
test_dataset = CustomDataset(test_datas, test_labels)


# DataLoaderの作成
data_loader = DataLoader(dataset=custom_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=100, shuffle=True)


class mlp_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.LSTM(20, 16, batch_first=True)
        self.fc3 = nn.Linear(16, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):

        x,_ = self.fc2(x)
        x = torch.sigmoid(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = F.relu(x)


        x = self.fc3(x)
        return x
    

def train(model, train_loader):
    # 今は学習時であることを明示するコード
    model.train()
    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
    for batch in train_loader:
        batch_imgs=batch['datas']
        batch_labels=batch['label']
        batch_imgs = batch_imgs.reshape(-1, 20)
        batch_labels = batch_labels.reshape(-1, 1)
        outputs = model(batch_imgs)  # 順伝播
        model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
        # print(outputs)
        # print(batch_labels)
        loss = model.criterion(outputs, batch_labels)  # 損失を計算
        loss.backward()  # 逆伝播で勾配を計算
        model.optimizer.step()  # 最適化




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
        batch_imgs = batch_imgs.reshape(-1, 20)
        batch_labels = batch_labels.reshape(-1, 1)
        outputs = model(batch_imgs)  # 順伝播
        loss = model.criterion(outputs, batch_labels)  # 損失を計算
       
        batch_size = len(batch_labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            if(total_data_len%100==0):
                #print("datas:"+str(batch_imgs[i].item())+"  out:"+str(outputs[i].item())+"  label:"+str(batch_labels[i].item())+"  difference:"+str((outputs[i].item() - batch_labels[i].item())**2))
                None
            total_data_len += 1  # 全データ数を集計
            if (outputs[i].item() - batch_labels[i].item())**2 < 10:
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