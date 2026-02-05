#モジュールのインポート
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch import utils
from torchvision import datasets
import torchvision.transforms as transforms


trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())  # 学習用データセット
train_loader = utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)  # ミニバッチごとにデータを纏める(学習時にはshuffle=True)

testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())  # 検証用データセット
test_loader = utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)  # ミニバッチごとにデータを纏める(学習時にはshuffle=False)


class mlp_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 10)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    

def train(model, train_loader):
    # 今は学習時であることを明示するコード
    model.train()

    # 正しい予測数、損失の合計、全体のデータ数を数えるカウンターの0初期化
    total_correct = 0
    total_loss = 0
    total_data_len = 0

    # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
    for batch_imgs, batch_labels in train_loader:
        batch_imgs = batch_imgs.reshape(-1, 28*28*1)  # 画像データを1次元に変換
        labels = torch.eye(10)[batch_labels]  # 正解ラベルをone-hotベクトルへ変換

        outputs = model(batch_imgs)  # 順伝播
        model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
        loss = model.criterion(outputs, labels)  # 損失を計算
        loss.backward()  # 逆伝播で勾配を計算
        model.optimizer.step()  # 最適化
       
        # ミニバッチごとの正答率と損失を求める
        _, pred_labels = torch.max(outputs, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
        batch_size = len(batch_labels)  # バッチサイズの確認
        for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
            total_data_len += 1  # 全データ数を集計
            if pred_labels[i] == batch_labels[i]:
                total_correct += 1 # 正解のデータ数を集計
        total_loss += loss.item()  # 全損失の合計

    # 今回のエポックの正答率と損失を求める
    accuracy = total_correct/total_data_len*100  # 予測精度の算出
    loss = total_loss/total_data_len  # 損失の平均の算出
    return accuracy, loss

# モデルを宣言する
model = mlp_net()

# 学習させ、その結果を表示する
acc, loss = train(model, train_loader)
print(f'正答率: {acc}, 損失: {loss}')