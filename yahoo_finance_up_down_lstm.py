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
import matplotlib.pyplot as plt



# csvからimport
df = pd.read_csv('./65days.csv',  header=0)
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_datas=df.iloc[0:, 0:64].to_numpy().astype(np.float32)
all_labels=df.iloc[0:, 64].to_numpy().astype(np.float32)

for i in range(len(all_labels)):
    if(all_labels[i]>all_datas[i][63]):
        all_labels[i]=1 #上がる
    else:
        all_labels[i]=0 #下がる

all_datas = np.log(all_datas[:, 1:] / all_datas[:, :-1])
all_datas=torch.from_numpy(all_datas)
all_labels=torch.from_numpy(all_labels)
print(all_datas)


datas = all_datas[::2].to(my_device)
labels = all_labels[::2].to(my_device)


test_datas = all_datas[1::2].to(my_device)
test_labels = all_labels[1::2].to(my_device)
print("データインポート完了")



# カスタムデータセットの作成
class CustomDataset(Dataset):
    def __init__(self, datas, labels):
        self.datas = datas
        self.labels = labels

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        return self.datas[idx], self.labels[idx]



# データセットのインスタンスを作成
custom_dataset = CustomDataset(datas, labels)
test_dataset = CustomDataset(test_datas, test_labels)


# DataLoaderの作成
data_loader = DataLoader(dataset=custom_dataset, batch_size=4000, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4000, shuffle=True)


class mlp_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 64, num_layers=2, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]

        x = self.fc(x)
        x = F.relu(x)
        x=self.dropout(x)


        x = self.fc2(x)
        return x


def train(model, train_loader):
    model.train()

    for i in range(300):
        # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
        for batch_items, batch_labels  in train_loader:
            batch_items = batch_items.view(-1, 63, 1)
            batch_labels = batch_labels.long()
            outputs = model(batch_items)  # 順伝播
            model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
            loss = model.criterion(outputs, batch_labels)  # 損失を計算
            loss.backward()  # 逆伝播で勾配を計算
            model.optimizer.step()  # 最適化
        print(f"{i}周目終了")




def test(model, train_loader):
    model.eval()

    # 正しい予測数、損失の合計、全体のデータ数を数えるカウンターの0初期化
    total_correct = 0
    total_loss = 0
    total_data_len = 0

    for batch_items, batch_labels  in train_loader:
        batch_items = batch_items.view(-1, 63, 1)
        batch_labels = batch_labels.long()
        outputs = model(batch_items)  # 順伝播
        loss = model.criterion(outputs, batch_labels)  # 損失を計算
       
        # ミニバッチごとの正答率と損失を求める
        _, pred_labels = torch.max(outputs, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
        batch_size = len(batch_labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで

            outputs_softmax=softmax=F.softmax(outputs[i], dim=0)
            if outputs_softmax[0].item()>0.6 or outputs_softmax[1].item()>0.6:
                print(pred_labels[i])
                print(batch_labels[i])

                total_data_len += 1  # 全データ数を集計
                if pred_labels[i] == batch_labels[i]:
                    total_correct += 1 # 正解のデータ数を集計
                if(total_data_len%10000==0):
                    print("out:"+str(outputs_softmax)+"  label:"+str(batch_labels[i]))
                    if pred_labels[i] == batch_labels[i]:
                        print("正解")
                    else:
                        print("不正解")
        total_loss += loss.item()  # 全損失の合計

    #正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    loss = total_loss/total_data_len  # 損失の平均の算出
    return accuracy, loss



# モデルを宣言する
model = mlp_net().to(my_device)
model.load_state_dict(torch.load('up_down_model_weight_100.pth', map_location=my_device))

# 学習させ、その結果を表示する
test(model, test_loader)
torch.save(model.state_dict(), 'up_down_model_weight_100.pth')
# model_scripted = torch.jit.script(model)
# model_scripted.save('up_down_model_scripted_100.pth')

acc, loss = test(model, test_loader)
print(f'正答率: {acc}, 損失: {loss}')