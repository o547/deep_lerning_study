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
df = pd.read_csv("./ecg.csv", header=0)
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 140
all_datas = df.iloc[0:, 0:input_size].to_numpy().astype(np.float32)
all_labels = df.iloc[0:, input_size].to_numpy().astype(np.float32)

all_datas = torch.from_numpy(all_datas)
all_labels = torch.from_numpy(all_labels)
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
data_loader = DataLoader(dataset=custom_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=True)


class linear_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters())

    def forward(self, x):
        return self.net(x)


def train(model, train_loader):
    model.train()

    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
    for batch_items, batch_labels in train_loader:
        batch_labels = batch_labels.long()
        model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
        outputs = model(batch_items)  # 順伝播
        loss = model.criterion(outputs, batch_labels)  # 損失を計算
        loss.backward()  # 逆伝播で勾配を計算
        model.optimizer.step()  # 最適化


def test(model, train_loader):
    model.eval()

    # 正しい予測数、損失の合計、全体のデータ数を数えるカウンターの0初期化
    total_correct = 0
    total_loss = 0
    total_data_len = 0
    with torch.no_grad():
        for batch_items, batch_labels in train_loader:
            batch_labels = batch_labels.long()
            outputs = model(batch_items)  # 順伝播
            loss = model.criterion(outputs, batch_labels)  # 損失を計算

            # ミニバッチごとの正答率と損失を求める
            _, pred_labels = torch.max(outputs, axis=1)
            batch_size = len(batch_labels)  # バッチサイズの確認
            for i in range(
                batch_size
            ):  # データ一つずつループ,ミニバッチの中身出しきるまで
                total_data_len += 1  # 全データ数を集計
                if pred_labels[i] == batch_labels[i]:
                    total_correct += 1  # 正解のデータ数を集計

            total_loss += loss.item()  # 全損失の合計

    # 正答率と損失を求める
    accuracy = total_correct / total_data_len * 100  # 予測精度の算出
    loss = total_loss / total_data_len  # 損失の平均の算出
    return accuracy, loss


# モデルを宣言する
model = linear_net().to(my_device)
# model.load_state_dict(
#     torch.load("up_down_model_weight_100.pth", map_location=my_device)
# )

accs = []
# 学習させ、その結果を表示する
for i in range(200):
    train(model, data_loader)
    print(f"{i}周目終了")
    if i % 5 == 0:
        acc, loss = test(model, test_loader)
        print(f"accuracy: {acc}, loss: {loss}")
        accs.append(acc)


# model_scripted = torch.jit.script(model)
# model_scripted.save("up_down_model_scripted_linear_100")
# torch.save(model.state_dict(), "up_down_model_scripted_linear_weight_100")
plt.plot(accs)
plt.show()
