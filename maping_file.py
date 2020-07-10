from config import Config
import pandas as pd
import os
import pickle

def mapping_creativevalues(x):
    value=creative_dict.get(x,0)
    if value:
        return value
    else:
        return x

def mapping_advalues(x):
    value=ad_dict.get(x,0)
    if value:
        return value
    else:
        return x

def mapping_product_dictvalues(x):
    value=product_dict.get(x,0)
    if value:
        return value
    else:
        return x

def mapping_advertiservalues(x):
    value=advertiser_dict.get(x,0)
    if value:
        return value
    else:
        return x



if __name__ == '__main__':
    config =Config()
    # if not os.path.exists(config.data_processed + 'merge.csv'):
    # print('Merge File ...')
    """加载训练集的文件"""
    click_log_df = pd.read_csv(config.data_path + 'train_preliminary/click_log.csv')
    ad_df = pd.read_csv(config.data_path + 'train_preliminary/ad.csv')
    """加载测试集的文件"""
    click_log_test = pd.read_csv(config.data_path + 'test/click_log.csv')
    ad_test = pd.read_csv(config.data_path + 'test/ad.csv')
    merge_test = pd.merge(click_log_test, ad_test, how='left', on='creative_id')

    merge_df = pd.merge(click_log_df, ad_df, how='left', on='creative_id')
    merge_df = merge_df.append(merge_test)
    #     merge_df.to_csv(config.data_processed + 'merge.csv', index=False)
    # else:
    #     merge_df = pd.read_csv(config.data_processed + 'merge.csv')
    """把\\N变成0"""
    merge_df = merge_df.replace('\\N','0')

    """转成相同类型"""
    print('Trans_type')
    merge_df['product_category']=merge_df['product_category'].astype('int')
    merge_df['industry'] = merge_df['industry'].astype('int')
    merge_df['ad_id']=merge_df['ad_id'].astype('int')
    merge_df['product_id']=merge_df['product_id'].astype('int')
    merge_df['advertiser_id']=merge_df['advertiser_id'].astype('int')
    merge_df['creative_id']=merge_df['creative_id'].astype('int')

    merge_df['industry'] = merge_df['industry'] + 18+1
    merge_df['product_id']=merge_df['product_id'] + 18+1+335+1
    merge_df['advertiser_id'] = merge_df['advertiser_id'] + 18+1+335+1+44314+1
    merge_df['ad_id'] = merge_df['ad_id'] + 18+1+335+1+44313+1+62965+1
    merge_df['creative_id'] = merge_df['creative_id'] + 18+1+335+1+44313+1+62965+1+3812202+1
    #
    merge_df.to_csv(config.ori_path + 'merge_ori.csv',index=False)

    # """筛选语料出现频次，生成字典映射某些id与值的关系"""
    # """频次小于128的creative_id映射到频次"""
    # """进统计，有共用的数字，所以映射值不放入同一个字典"""
    # if not os.path.exists(config.data_processed + 'creative_dict.pickle'):
    #     creative_ss = merge_df['creative_id'].value_counts()
    #     creative_dict = dict([(i, j+8365554) for i, j in creative_ss.to_dict().items() if j <= 256])  # 合并两个表后，28684
    #     print('不映射的字典长度creative：',len(creative_ss)-len(creative_dict))
    #     ad_ss = merge_df['ad_id'].value_counts()
    #     ad_dict = dict([(i, j+8365554+1+256) for i, j in ad_ss.to_dict().items() if j <= 256])  # 合并两个表后，28849
    #     print('不映射的字典长度ad：', len(ad_ss) - len(ad_dict))
    #     product_ss = merge_df['product_id'].value_counts()
    #     product_dict = dict([(i, j+8365554+1+512+1) for i, j in product_ss.to_dict().items() if j <= 128])  #5180
    #     print('不映射的字典长度product：', len(product_ss) - len(product_dict))
    #     advertiser_ss = merge_df['advertiser_id'].value_counts()
    #     advertiser_dict = dict([(i, j+8365554+1+512+1+128+1) for i, j in advertiser_ss.to_dict().items() if j <= 128]) #14712
    #     print('不映射的字典长度advertiser：', len(advertiser_ss) - len(advertiser_dict))
    #     with open(config.data_processed + 'creative_dict.pickle', 'wb') as mf:
    #         pickle.dump(creative_dict, mf)
    #     with open(config.data_processed + 'ad_dict.pickle', 'wb') as mf:
    #         pickle.dump(ad_dict, mf)
    #     with open(config.data_processed + 'product_dict.pickle', 'wb') as mf:
    #         pickle.dump(product_dict, mf)
    #     with open(config.data_processed + 'advertiser_dict.pickle', 'wb') as mf:
    #         pickle.dump(advertiser_dict, mf)
    # else:
    #     with open(config.data_processed + 'creative_dict.pickle', 'rb') as mf:
    #         creative_dict = pickle.load(mf)
    #     with open(config.data_processed + 'ad_dict.pickle', 'rb') as mf:
    #         ad_dict = pickle.load(mf)
    #     with open(config.data_processed + 'product_dict.pickle', 'rb') as mf:
    #         product_dict = pickle.load(mf)
    #     with open(config.data_processed + 'advertiser_dict.pickle', 'rb') as mf:
    #         advertiser_dict = pickle.load(mf)
    # """creative处理后喂进bert的独立字字长为  18+326+28684 +5180+14712 = 48920独立类别"""
    # """ad处理后喂进bert的独立字字长为  18+326+28849 +5180+14712 = 49085独立类别"""
    #
    # merge_df['creative_id'] = merge_df['creative_id'].apply(mapping_creativevalues)
    # merge_df['product_id'] = merge_df['product_id'].apply(mapping_product_dictvalues)
    # merge_df['advertiser_id'] = merge_df['advertiser_id'].apply(mapping_advertiservalues)
    # merge_df['ad_id'] = merge_df['ad_id'].apply(mapping_advalues)
    # merge_df.to_csv(config.data_processed+'merge_mapping.csv',index=False)
