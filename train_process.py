from config import Config
import pandas as pd
import os

config=Config()
str_flag = 'ad'
print(str_flag)

file_path = '/home/none404/hm/Tencent_ads/data/processed_xuan/'
file_list = os.listdir(file_path)
flag = 1
for file in file_list:
    if 'csv' in file and str_flag in file:
        print(file)
        if flag:
            flag = 0
            df = pd.read_csv(file_path + file)
        else:

            temp = pd.read_csv(file_path+file)
            df = df.append(temp)

print(len(df))

train_df = df[df['age']!=0]
test_df = df[df['age']==0]
print('TrainSet',train_df.shape)
print('TestSet',test_df.shape)

train_df.to_csv(config.finetune_xuan+'{}_data/'.format(str_flag)+'train.csv',index=False)
test_df.to_csv(config.finetune_xuan+'{}_data/'.format(str_flag)+'test.csv',index=False)

train_df['sequence_len']=train_df['text'].apply(lambda x:x.split(' ').__len__())
test_df['sequence_len']=test_df['text'].apply(lambda x:x.split(' ').__len__())

print(train_df['sequence_len'].describe())
print(test_df['sequence_len'].describe())

"""
count    900000.000000
mean         89.988656
std          68.180660
min          13.000000
25%          49.000000
50%          70.000000
75%         107.000000
max       20784.000000
Name: sequence_len, dtype: float64
count    1000000.000000
mean          90.131457
std          148.564116
min           14.000000
25%           49.000000
50%           70.000000
75%          107.000000
max       122783.000000
Name: sequence_len, dtype: float64
"""