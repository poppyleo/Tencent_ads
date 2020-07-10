from gensim.models import Word2Vec
from config import Config
import pandas as pd
import logging
import gensim
from tqdm import tqdm
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from pandarallel import pandarallel

pandarallel.initialize()

config = Config()
file_path = config.corpus_path


def category_t(x):
    x = x.replace(' 天 ', '/')
    x = x.replace(' 地 ', '/')
    x = x.replace(' 玄 ', '/')
    x = x.replace(' 黄 ', '/')
    x = x.replace(';', '/')
    x_list = x.split('/')
    return x_list[0]


def industry_t(x):
    x = x.replace(' 天 ', '/')
    x = x.replace(' 地 ', '/')
    x = x.replace(' 玄 ', '/')
    x = x.replace(' 黄 ', '/')
    x = x.replace(';', '/')
    x_list = x.split('/')
    return x_list[1]


def product_t(x):
    x = x.replace(' 天 ', '/')
    x = x.replace(' 地 ', '/')
    x = x.replace(' 玄 ', '/')
    x = x.replace(' 黄 ', '/')
    x = x.replace(';', '/')
    x_list = x.split('/')
    return x_list[2]


def advertiser_t(x):
    x = x.replace(' 天 ', '/')
    x = x.replace(' 地 ', '/')
    x = x.replace(' 玄 ', '/')
    x = x.replace(' 黄 ', '/')
    x = x.replace(';', '/')
    x_list = x.split('/')
    return x_list[3]


def creative_t(x):
    x = x.replace(' 天 ', '/')
    x = x.replace('  地  ', '/')
    x = x.replace(' 玄 ', '/')
    x = x.replace(' 黄 ', '/')
    x = x.replace(';', '/')
    x_list = x.split('/')
    return x_list[4]


### 做embedding 这里采用word2vec 可以换成其他例如（glove词向量）
def trian_save_word2vec(docs, window, embed_size=256, save_name='w2v.txt', split_char=' '):
    '''
    输入
    docs:输入的文本列表
    embed_size:embed长度
    save_name:保存的word2vec位置

    输出
    w2v:返回的模型
    '''
    input_docs = []
    for i in docs:
        input_docs.append(i.split(split_char))
    logging.basicConfig(
        format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
    w2v = Word2Vec(input_docs, size=embed_size, sg=1, window=window, seed=2020, workers=32, min_count=1, iter=10)
    w2v.wv.save_word2vec_format(save_name)
    print("w2v model done")
    return w2v





if __name__ == '__main__':
    train_df = pd.read_csv(file_path + 'train.csv', encoding='utf_8_sig')
    test_df = pd.read_csv(file_path + 'test.csv', encoding='utf_8_sig')

    merge_df = train_df.append(test_df)[:1000]
    del train_df, test_df
    embed_size = 256
    print('find_data...')
    category_list = merge_df['text'].parallel_apply(category_t).to_list()

    # industry_list = merge_df['text'].apply(industry_t).to_list()
    # product_list = merge_df['text'].apply(product_t).to_list()
    # advertiser_list = merge_df['text'].apply(advertiser_t).to_list()
    # creative_list = merge_df['text'].apply(creative_t).to_list()
    print('开始序列化')
    # 适当修整长度
    max_len = 14
    x1, index_1 = set_tokenizer(category_list, split_char=' ', max_len=14)
    # x2, index_2 = set_tokenizer(industry_list, split_char=' ', max_len=36)
    # x3, index_3 = set_tokenizer(product_list, split_char=' ', max_len=48)
    # x4, index_4 = set_tokenizer(product_list, split_char=' ', max_len=48)
    # x5, index_5 = set_tokenizer(creative_list, split_char=' ', max_len=80)
    print('序列化完成')
    trian_save_word2vec(category_list, window=int(max_len / 2), save_name=config.model_path + 'w2v_model/cate_256.txt',
                        split_char=' ')

    #### 得到emb矩阵
    emb1 = get_embedding_matrix(index_1, Emed_path=config.model_path + 'w2v_model/cate_256.txt')

    # emb3 = get_embedding_matrix(index_3, Emed_path='w2v_model/w2v_300.txt')
