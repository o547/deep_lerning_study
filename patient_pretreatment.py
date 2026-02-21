import pandas as pd
import numpy as np
import datetime

df = pd.read_csv("./KaggleV2-May-2016.csv", header=0)
df = df.fillna(0)
df = df.drop(["PatientId", "AppointmentID", "Neighbourhood"], axis=1)
df = df.replace({"F": 0, "M": 1, "No": 0, "Yes": 1})

for i in range(len(df)):
    dt1 = datetime.datetime.strptime(df["ScheduledDay"][i], "%Y-%m-%dT%H:%M:%SZ")
    dt2 = datetime.datetime.strptime(df["AppointmentDay"][i], "%Y-%m-%dT%H:%M:%SZ")
    df.loc[i, "ScheduledDay"] = (dt2 - dt1).days+1
    df.loc[i, "AppointmentDay"] = dt2.strftime("%a")
df=df.rename(columns={'ScheduledDay': 'ReservationDateDifference'})
df = pd.get_dummies(df, columns=["AppointmentDay","No-show"],dtype = 'uint8')
# df_ans=df["No-show"]
# df=df.drop("No-show",axis=1)
# df.insert(df.shape[1],"No-show",df_ans)
print(df)
df.to_csv('patient.csv',header=True,index=False)
