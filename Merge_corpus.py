# import sys
# sys.path.extend('/home/none404/hm/Tencent_ads')
from config import Config
import pandas as pd
import numpy as np

config =Config()
import os
#
file_path ='/home/none404/hm/Tencent_ads/data/processed_xuan/'
out_path ='/home/none404/hm/Tencent_ads/finetune_xuan/'
file_list  =  os.listdir(file_path)
text_list = []
for file_name in file_list:
    if 'txt' in file_name and 'creative' in file_name:
        with open(file_path+file_name,'r') as f:
            for text in f.readlines():
                text_list.append(text)

with open(out_path+'Corpus_creative/all_corpus.txt','w') as f_all:
    f_all.writelines(text_list)
text_len = [i.split(' ').__len__() for i in text_list]


print('creative语料长度分布为：')
print(pd.Series(text_len).describe())
len_des = np.array(text_len)
len_des = np.sort(len_des)
print('95%覆盖的长度：',len_des[int(len(len_des)*0.95)])

text_list = []
for file_name in file_list:
    if 'txt' in file_name and 'ad' in file_name:
        with open(file_path+file_name,'r') as f:
            for text in f.readlines():
                text_list.append(text)

with open(out_path+'Corpus_ad/all_corpus.txt','w') as f_all:
    f_all.writelines(text_list)
text_len = [i.split(' ').__len__() for i in text_list]

print('ad语料长度分布为：')
print(pd.Series(text_len).describe())
len_des = np.array(text_len)
len_des = np.sort(len_des)
print('95%覆盖的长度：',len_des[int(len(len_des)*0.95)])



"""
creative语料长度分布为：
count    1.900000e+06
mean     9.006381e+01
std      1.175519e+02
min      1.300000e+01
25%      4.900000e+01
50%      7.000000e+01
75%      1.070000e+02
max      1.227830e+05
dtype: float64
95%覆盖的长度： 210
 ad语料长度分布为：
count    1.900000e+06
mean     9.006381e+01
std      1.175519e+02
min      1.300000e+01
25%      4.900000e+01
50%      7.000000e+01
75%      1.070000e+02
max      1.227830e+05
dtype: float64
95%覆盖的长度： 210

"""