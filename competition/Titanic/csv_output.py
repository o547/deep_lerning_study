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
    all_datas = df.iloc[0:, 1 : input_size+1].to_numpy().astype(np.float32)

    datas = torch.from_numpy(all_datas).to(my_device)


    # モデルを宣言する
    model = torch.jit.load(model_path)
    title=np.array(["PassengerId","Survived"])
    table=title.reshape(1, -1)

    for i in range(datas.size()[0]):
        out=model(datas[i]).to("cpu").detach().numpy()
        # print(df.loc[i,"PassengerId"])
        column = np.array([df.loc[i,"PassengerId"] ,np.argmax(out)])
        column = column.reshape(1, -1)   # (1,n) に変換
        # print(table,column)
        # table=np.append(table, out, axis=0)
        table = np.concatenate([table,column], axis=0)
    df_table=pd.DataFrame(table)
    #print(df_table)
    # np.savetxt("out.csv",table,delimiter=",",fmt="%.5f")
    df_table.to_csv(file_path.replace(".csv","_out.csv"),header=False,index=False,encoding=character_encoding)



learn(file_path="./test2.csv", model_path="./train2.pth", file_name="sample", input_size=11, output_size=2)

