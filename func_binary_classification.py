def learn(file_path, file_name, epoch_size, input_size, mid1_size, mid2_size, output_size=1):
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    from torch import utils
    from torch.utils.data import DataLoader, Dataset

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    my_device="cpu"
    
    #GPUをOS側で設定済みであればコメントアウト
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class CustomDataset(Dataset):
        def __init__(self, datas, labels):
            self.datas = datas
            self.labels = labels

        def __len__(self):
            return len(self.datas)

        def __getitem__(self, idx):
            return self.datas[idx], self.labels[idx]
    try:
        df = pd.read_csv(file_path, header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, header=0, encoding="shift_jis")

    df = df.fillna(0)

    all_datas = df.iloc[0:, 0 : input_size].to_numpy().astype(np.float32)
    all_labels = df.iloc[0:, input_size : (input_size + 1)].to_numpy().astype(np.float32)
    idx_datas = np.arange(len(all_datas))
    idx_labels = np.arange(len(all_labels))

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
    elif (len(all_datas)<1000):
        batch_size=100
    else:
        batch_size=1000

    # DataLoaderの作成
    data_loader = DataLoader(dataset=custom_dataset, batch_size=4096, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=4096, shuffle=True)

    class mlp_net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_size, mid1_size)
            self.fc2 = nn.Linear(mid1_size, mid2_size)
            self.fc2_5 = nn.Linear(mid2_size, mid2_size)
            self.fc3 = nn.Linear(mid2_size, 2)
            weights = torch.tensor([0.6265, 2.4751])
            self.criterion = nn.CrossEntropyLoss(weight = weights)
            self.optimizer = optim.Adam(self.parameters())
            self.dropout = nn.Dropout(0.05)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)

            x = self.fc2(x)
            x = F.relu(x)

            # x=self.dropout(x)

            for i in range(8):
                x = self.fc2_5(x)
                x = F.relu(x)

            x = self.fc3(x)
            return x

    def train(model, train_loader):
        model.train()
        for i in range(1):
            # ミニバッチごとにループさせる,train_loaderの中身を出し切ったら1エポックとなる
            for batch_items, batch_labels in train_loader:
                batch_labels = batch_labels.view(-1).long()
                outputs = model(batch_items)  # 順伝播
                model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
                loss = model.criterion(outputs, batch_labels)  # 損失を計算
                loss.backward()  # 逆伝播で勾配を計算
                model.optimizer.step()  # 最適化


    def test(model, train_loader):
        model.eval()

        # 正しい予測数、損失の合計、全体のデータ数を数えるカウンターの0初期化
        total_correct = 0
        total_loss = 0
        total_data_len = 0
        total_detection = 0

        for batch_items, batch_labels  in train_loader:
            batch_labels = batch_labels.view(-1).long()
            outputs = model(batch_items)  # 順伝播
            model.optimizer.zero_grad()  # 勾配を初期化（前回のループ時の勾配を削除）
            loss = model.criterion(outputs, batch_labels)  # 損失を計算
            loss.backward()  # 逆伝播で勾配を計算
        
            # ミニバッチごとの正答率と損失を求める
            _, pred_labels = torch.max(outputs, axis=1)  # outputsから必要な情報(予測したラベル)のみを取り出す。
            batch_size = len(batch_labels)  # バッチサイズの確認
            for i in range(batch_size):  # データ一つずつループ,ミニバッチの中身出しきるまで
                    outputs_softmax=F.softmax(outputs[i], dim=0)
                    if outputs_softmax[0].item()>0.65 or outputs_softmax[1].item()>0.65:
                        total_detection+=1
                        total_data_len += 1  # 全データ数を集計
                        if pred_labels[i] == batch_labels[i]:
                            total_correct += 1 # 正解のデータ数を集計
            total_loss += loss.item()  # 全損失の合計
        try:
            accuracy = total_correct / total_data_len * 100  # 予測精度の算出
            loss = total_loss / total_data_len  # 損失の平均の算出
        except ZeroDivisionError:
            accuracy = -0.1
            loss = -0.1
        return total_detection, accuracy, loss


    model = mlp_net()
    model = model.to(my_device)
    # 重みをロード
    # model.load_state_dict(torch.load(file_path.replace(".csv","4_weight.pth"), map_location=my_device))
    # モデルをロード
    # model = torch.jit.load(file_path.replace(".csv",".pth"), map_location=my_device)


    accs=[]
    losses=[]
    for i in range(epoch_size):
        train(model, data_loader)
        if(i%5==0):
            detection, acc, loss = test(model, test_loader)
            accs.append(acc)
            losses.append(loss)
            print(f"{i}周目　正答率: {acc}, 損失: {loss}, 検知数：{detection}")
            if(i%50==0):
                torch.save(model.state_dict(), file_path.replace(".csv","4weight.pth"))
                model_scripted = torch.jit.script(model)
                model_scripted.save(file_path.replace(".csv","4.pth"))
    plt.plot(accs)

    torch.save(model.state_dict(), file_path.replace(".csv","4_weight.pth"))
    model_scripted = torch.jit.script(model)
    model_scripted.save(file_path.replace(".csv","4.pth"))
    plt.show()



learn("./patient.csv", "patient", 200, 15, 256, 128)