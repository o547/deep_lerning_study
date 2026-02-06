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
# import scipy.stats



# csvからimport
df = pd.read_csv('./21days.csv',  header=0)

all_datas=df.iloc[0:, 0:20].to_numpy().astype(np.float32)
all_labels=df.iloc[0:, 20].to_numpy().astype(np.float32)
print(all_datas)


for i in range(len(all_labels)):
    if(all_labels[i]>all_datas[i][19]):
        all_labels[i]=1 #上がる
    else:
        all_labels[i]=0 #下がる

all_datas = np.log(all_datas[:, 1:] / all_datas[:, :-1])
print(all_datas)


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
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.fc1 = nn.Linear(19, 16)
        self.fc2 = nn.Linear(16, 16)

        self.fc = nn.Linear(16, 2)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # _, (h_n, _) = self.lstm(x)
        # x = h_n[-1]
        x = self.fc(x)
        return x

def train(model, train_loader):
    # 今は学習時であることを明示するコード
    model.train()
    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
    for i in range(10):
        for batch in train_loader:
            batch_items=batch['datas']
            # batch_items = batch_items.reshape(-1, 19, 1)
            batch_labels=batch['label']
            batch_labels = batch_labels.long()
            outputs = model(batch_items)  # 順伝播
            model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
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
        batch_items=batch['datas']
        # batch_items = batch_items.reshape(-1, 19, 1)
        batch_labels=batch['label']
        batch_labels = batch_labels.long()
        outputs = model(batch_items)  # 順伝播
        loss = model.criterion(outputs, batch_labels)  # 損失を計算
       
        # ミニバッチごとの正答率と損失を求める
        _, pred_labels = torch.max(outputs, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
        batch_size = len(batch_labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred_labels[i] == batch_labels[i]:
                total_correct += 1 # 正解のデータ数を集計
            if(total_data_len%10000==0):
                print("data:"+str(batch_items[i])+"  out:"+str(outputs[i])+"  label:"+str(batch_labels[i]))
                if pred_labels[i] == batch_labels[i]:
                    print("正解")
                else:
                    print("不正解")
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