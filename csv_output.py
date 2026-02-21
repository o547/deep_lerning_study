def learn(file_path, model_path, file_name, input_size, output_size):
    import torch
    from torch import nn, optim
    from torch.nn import functional as F
    from torch import utils

    import pandas as pd
    import numpy as np
    from torch.utils.data import DataLoader, Dataset

    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    character_encoding="utf-8"
    try:
        df = pd.read_csv(file_path, header=0)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, header=0, encoding="shift_jis")
        character_encoding="shift_jis"

    df = df.fillna(0)
    all_datas = df.iloc[0:, 0 : input_size].to_numpy().astype(np.float32)

    datas = torch.from_numpy(all_datas).to(my_device)


    # モデルを宣言する
    model = torch.jit.load(model_path)
    table = df.columns.to_numpy()
    table = table.reshape(1, -1)   # (1,5) に変換

    for i in range(datas.size()[0]):
        in_data=datas[i].to("cpu").numpy()
        out=model(datas[i]).to("cpu").detach().numpy()
        column=np.concatenate([in_data,out])
        column = column.reshape(1, -1)   # (1,5) に変換
        table=np.append(table, column, axis=0)
        #table = np.concatenate([table,column], axis=0)
    df_table=pd.DataFrame(table)
    #print(df_table)
    # np.savetxt("out.csv",table,delimiter=",",fmt="%.5f")
    df_table.to_csv(file_path.replace(".csv","_out.csv"),header=False,index=False,encoding=character_encoding)



learn(file_path="./sample.csv", model_path="./sample.pth", file_name="sample", input_size=3, output_size=2)

