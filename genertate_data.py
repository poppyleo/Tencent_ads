import pandas as pd
import pickle
from config import Config
import time
from pandarallel import pandarallel
pandarallel.initialize()

def join_id(ss):
    creative_ss = ss.groupby('creative_id')['click_times'].sum()
    product_ss = ss.groupby('product_id')['click_times'].sum()
    category_ss = ss.groupby('product_category')['click_times'].sum()
    advertiser_ss = ss.groupby('advertiser_id')['click_times'].sum()
    industry_ss = ss.groupby('industry')['click_times'].sum()

    creative_ss = creative_ss[creative_ss != -1]
    product_ss = product_ss[product_ss != -1]
    category_ss = category_ss[category_ss != -1]
    advertiser_ss = advertiser_ss[advertiser_ss != -1]
    industry_ss = industry_ss[industry_ss != -1]

    creative_text = ' '.join([str(i) for i in list(creative_ss.keys())])
    product_text = ' '.join([str(i) for i in list(product_ss.keys())])
    category_text = ' '.join([str(i) for i in list(category_ss.keys())])
    advertiser_text = ' '.join([str(i) for i in list(advertiser_ss.keys())])
    industry_text = ' '.join([str(i) for i in list(industry_ss.keys())])

    creative_times = ' '.join([str(i) for i in list(creative_ss.values)])
    product_times = ' '.join([str(i) for i in list(product_ss.values)])
    category_times = ' '.join([str(i) for i in list(category_ss.values)])
    advertiser_times = ' '.join([str(i) for i in list(advertiser_ss.values)])
    industry_times = ' '.join([str(i) for i in list(industry_ss.values)])

    all_text = category_text + ' 天 ' + industry_text + ' 地 ' + product_text + ' 玄 ' + advertiser_text + ' 黄 ' + creative_text
    all_times = category_times + ' 1 ' + industry_times + ' 1 ' + product_times + ' 1 ' + advertiser_times + ' 1 ' + creative_times

    return all_text + ';' + all_times


def remove_buket(row):
    for remove_char in remove_list:
        row = row.replace(remove_char, '')
    return row


if __name__ == '__main__':
    start = time.time()
    # with open("/mnt/inspurfs/user-fs/lifeipeng/data/corpus/128_4pid_4aid/utils_delete_list.pickle", 'rb') as f:
    # with open("/home/wangzhili/lei/Tencent/data/data/utils_delete_list.pickle", 'rb') as f:
    #     remove_list = pickle.load(f)
    print('loading data...')
    config = Config()
    file_path = config.merge_path
    merge_file = pd.read_csv(file_path + 'merge_ori.csv')
    # label_df = pd.read_csv(config.ori_path + 'train_preliminary/user.csv')
    label_df = pd.read_csv(file_path + 'user.csv')
    # print('Deal bucket...')
    # """删除所有桶"""
    # bool_df = merge_file[['creative_id', 'click_times', 'ad_id', 'product_id',
    #                       'product_category', 'advertiser_id', 'industry']].isin(remove_list)
    #
    # merge_file[bool_df] = -1  # 桶子映射成-1

    print('generate data...')
    res_series = merge_file.groupby(['user_id'])[['creative_id', 'click_times', 'ad_id', 'product_id',
                                                  'product_category', 'advertiser_id', 'industry']].parallel_apply(join_id)
    print('writting...')
    creative_corpus = [' '.join(i.split(';')[0]) + '\n' for i in res_series]
    with open(config.corpus_path + 'creative_corpus.txt', 'w', encoding='utf-8') as f:
        f.writelines(creative_corpus)

    res_df = res_series.reset_index(level=None, drop=False, name=None, inplace=False)  # 变成dataframe
    labeled_df = pd.merge(res_df, label_df, on='user_id', how='left')

    labeled_df.fillna(-1, inplace=True)
    labeled_df['age'] = labeled_df['age'].astype(int)
    labeled_df['gender'] = labeled_df['gender'].astype(int)
    labeled_df.rename(columns={0: 'text'},inplace=True)
    train_df = labeled_df[labeled_df['age'] != -1]
    test_df = labeled_df[labeled_df['age'] == -1]
    train_df.to_csv(config.corpus_path + 'train.csv', index=False)
    test_df.to_csv(config.corpus_path + 'test.csv', index=False)

    labeled_df['len'] = labeled_df['text'].parallel_apply(lambda x: len(x.split(' ')) / 2)
    print(labeled_df['len'].describe())
    end = time.time()
    print('运行时长为', end - start)

    ####
    # generate
    # data...
    # writting...
    # count
    # 1.900000e+06
    # mean
    # 8.679945e+01
    # std
    # 8.917776e+01
    # min
    # 8.500000e+00
    # 25 % 4.750000e+01
    # 50 % 6.750000e+01
    # 75 % 1.035000e+02
    # max
    # 7.876250e+04
    # Name: len, dtype: float64
    # 运行时长为
    # 2273.4839239120483