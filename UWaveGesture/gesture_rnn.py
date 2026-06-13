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
df = pd.read_csv("./UWaveGestureLibraryAll_TRAIN.csv", header=0)
# my_device = "cpu"
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sequence_length = 945
all_datas = df.iloc[0:, 0:sequence_length].to_numpy().astype(np.float32)
all_labels = df.iloc[0:, sequence_length].to_numpy().astype(np.float32)

all_datas = torch.from_numpy(all_datas)
all_labels = torch.from_numpy(all_labels)
idx_datas = np.arange(len(all_datas))
idx_labels = np.arange(len(all_labels))

all_labels = all_labels % 8


datas = all_datas[idx_datas % 4 != 0]
labels = all_labels[idx_labels % 4 != 0]

test_datas = all_datas[idx_datas % 4 == 0]
test_labels = all_labels[idx_labels % 4 == 0]
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
data_loader = DataLoader(dataset=custom_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)


class lstm_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,  # 1日あたりの特徴量数
            hidden_size=32,  # 隠れ状態の次元
            num_layers=1,  # LSTM層の数
            batch_first=True,  # 入力を (batch, seq_len, feature) にする
        )
        self.fc = nn.Linear(32, 8)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        # lstm_out, (h_n, c_n) = self.lstm(x)
        rnn_out, h_n = self.rnn(x)

        # 最後の日の出力だけ使う
        last_out = rnn_out[:, -1, :]
        x = self.fc(last_out)

        return x


def train(model, train_loader):
    model.train()

    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
    for batch_items, batch_labels in train_loader:
        batch_items = batch_items.to(my_device)
        batch_labels = batch_labels.to(my_device)
        batch_items = batch_items.unsqueeze(-1)
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
    precision_denominator = 0
    precision_molecule = 0
    recall_denominator = 0
    recall_molecule = 0

    with torch.no_grad():
        for batch_items, batch_labels in train_loader:
            batch_items = batch_items.to(my_device)
            batch_labels = batch_labels.to(my_device)
            batch_items = batch_items.unsqueeze(-1)
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
            # # 適合率
            # if pred_labels[i] == 1:
            #     precision_denominator += 1
            #     if batch_labels[i] == 1:
            #         precision_molecule += 1
            # # 再現率
            # if batch_labels[i] == 1:
            #     recall_denominator += 1
            #     if pred_labels[i] == 1:
            #         recall_molecule += 1

        total_loss += loss.item()  # 全損失の合計
    try:
        # 予測精度の算出
        accuracy = total_correct / total_data_len * 100
        # precision = precision_molecule / precision_denominator * 100
        # recall = recall_molecule / recall_denominator * 100

        loss = total_loss / total_data_len  # 損失の平均の算出
    except ZeroDivisionError:
        accuracy = -0.1
        precision = -0.1
        recall = -0.1
        loss = -0.1
    return total_correct, accuracy, 0, 0, 0


# モデルを宣言する
model = lstm_net().to(my_device)
# model.load_state_dict(torch.load("rnn_weight.pth", map_location=my_device))

accs = []
for i in range(1000):
    train(model, data_loader)
    if i % 10 == 0:
        total_correct, accuracy, precision, recall, loss = test(model, test_loader)
        print(f"{i}周目 accuracy: {accuracy}, 正解数：{total_correct}")
        accs.append(accuracy)

# model_scripted = torch.jit.script(model)
# model_scripted.save("rnn_model.pth")
torch.save(model.state_dict(), "rnn_weight.pth")
plt.plot(accs)
plt.show()
