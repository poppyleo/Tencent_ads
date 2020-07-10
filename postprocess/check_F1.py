import os
import sys
sys.path.append("/home/wangzhili/lei/Tencent/")
from config import Config
config = Config()

folder_path = config.save_model
for folder_l1 in os.listdir(folder_path):
    for folder_l2 in os.listdir(folder_path + folder_l1):
        result_list = folder_path + folder_l1 + '/' + folder_l2 + '/' + 'checkpoint'
        if os.path.exists(folder_path + folder_l1 + '/' + folder_l2 + '/' + 'checkpoint'):
            max_acc = 0
            max_p = max_r = 0
            max_l = ''
            with open(result_list, encoding='utf-8') as file:
                for l in file.readlines():
                    line = l.strip().split('_')
                    p = float(line[-2]) + 1e-10  # f1
                    r = float(line[-1].split('-')[0]) + 1e-10  # acc
                    acc = p + r
                    if acc > max_acc:
                        max_acc = acc
                        max_p = p
                        max_r = r
                        max_l = l
            max_acc = round(max_acc,3)
            max_p = round(max_p,3)
            max_r = round(max_r,3)
            print(
                '{} {} :focal_r {} , age_r {}, gender_r {}'.format(folder_l1, '/'.join(max_l.split('"')[-2].split('/')[-2:]),max_acc, max_p, max_r))