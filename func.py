def learn(file_name, epoch_size, input_size, mid1_size, mid2_size, output_size):
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    from torch import utils
    from torchvision import datasets
    import torchvision.transforms as transforms

    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            return self.datas[idx], self.labels[idx]

    df = pd.read_csv("./" + file_name + ".csv", header=0)
    df = df.fillna(0)

    all_datas = df.iloc[0:, 0 : input_size].to_numpy().astype(np.float32)
    all_labels = df.iloc[0:, input_size : (input_size + output_size)].to_numpy().astype(np.float32)

    idx_datas = np.arange(len(all_datas))
    idx_labels = np.arange(len(all_labels))

    print(all_datas)


    datas = torch.from_numpy(all_datas[idx_datas % 4 != 0]).to(my_device)
    labels = torch.from_numpy(all_labels[idx_labels % 4 != 0]).to(my_device)

    test_datas = torch.from_numpy(all_datas[idx_datas % 4 == 0]).to(my_device)
    test_labels = torch.from_numpy(all_labels[idx_labels % 4 == 0]).to(my_device)



    # データセットのインスタンスを作成
    custom_dataset = CustomDataset(datas, labels)
    test_dataset = CustomDataset(test_datas, test_labels)

    batch_size=0
    if(len(all_datas)<10):
        batch_size=1
    elif (len(all_datas)<100):
        batch_size=10
    else:
        batch_size=100

    # DataLoaderの作成
    data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    class mlp_net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, mid1_size)
            self.fc2 = nn.Linear(mid1_size, mid2_size)
            self.fc3 = nn.Linear(mid2_size, output_size)
            self.criterion = nn.MSELoss()
            self.optimizer = optim.Adam(self.parameters())
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)

            x = self.fc2(x)
            x = F.relu(x)

            x = self.dropout(x)

            x = self.fc3(x)
            return x

    def train(model, train_loader):
        model.train()
        for i in range(epoch_size):
            # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
            for batch_items, batch_labels in train_loader:
                batch_labels = batch_labels.view(-1, 1)
                outputs = model(batch_items)  # 順伝播
                model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
                # print(outputs)
                # print(batch_labels)
                loss = model.criterion(outputs, batch_labels)  # 損失を計算
                loss.backward()  # 逆伝播で勾配を計算
                model.optimizer.step()  # 最適化


    def test(model, train_loader):
        model.eval()

        # 正しい予測数、損失の合計、全体のデータ数を数えるカウンターの0初期化
        total_correct = 0
        total_loss = 0
        total_data_len = 0

        # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
        for batch_items, batch_labels in train_loader:
            batch_labels = batch_labels.view(-1, 1)
            outputs = model(batch_items)  # 順伝播
            loss = model.criterion(outputs, batch_labels)  # 損失を計算

            batch_size = len(batch_labels)  # バッチサイズの確認
            for i in range(
                batch_size
            ):  # データ一つずつループ,ミニバッチの中身出しきるまで
                total_data_len += 1  # 全データ数を集計
                if (outputs[i].item() - batch_labels[i].item()) ** 2 < 9:
                    total_correct += 1  # 正解のデータ数を集計
            total_loss += loss.item()  # 全損失の合計

        # 今回のエポックの正答率と損失を求める
        accuracy = total_correct / total_data_len * 100  # 予測精度の算出
        loss = total_loss / total_data_len  # 損失の平均の算出
        return accuracy, loss

    # モデルを宣言する
    model = mlp_net()
    model = model.to(my_device)
    # 重みをロード
    # model.load_state_dict(torch.load('student_2_weight.pth', map_location=my_device))

    # datas=df.iloc[0, 0:19].to_numpy().astype(np.float32)
    # datas=torch.from_numpy(datas).to(my_device)
    # outputs = model(datas)
    # print(outputs)

    # 学習させ、その結果を表示する
    train(model, data_loader)
    acc, loss = test(model, test_loader)
    print(f"正答率: {acc}, 損失: {loss}")

    model_scripted = torch.jit.script(model)
    model_scripted.save(file_name + ".pth")
