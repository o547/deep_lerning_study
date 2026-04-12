import pandas as pd
import numpy as np
import datetime

train_data = pd.read_csv("./train.csv", header=0)
# print(train_data.head())
train_data = train_data.drop(["Name","Ticket","Cabin"], axis=1)

train_data["Embarked"] = train_data["Embarked"].fillna("S")
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())

train_data = pd.get_dummies(train_data, columns=["Pclass","Embarked"],dtype = 'uint8')
train_data = train_data.replace({"male": 0, "female": 1})
# print(train_data)

train_data.to_csv('train2.csv',header=True,index=False)
