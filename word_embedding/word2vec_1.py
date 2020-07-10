from gensim.models import Word2Vec
import sys
sys.path.append("/home/wangzhili/lei/Tencent")
from config import Config
import pandas as pd
import logging

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
    """category"""  # TODO
    flag= "category"
    train_df = pd.read_csv(file_path + 'train.csv', encoding='utf_8_sig')
    test_df = pd.read_csv(file_path + 'test.csv', encoding='utf_8_sig')
    merge_df = train_df.append(test_df)
    print(merge_df.shape)
    del train_df, test_df
    max_len = 14
    embed_size = 256
    print('find_data...')
    category_list = merge_df['text'].parallel_apply(category_t).to_list()
    print(len(category_list))
    # print('开始序列化')
    # # 适当修整长度
    # x1, index_1 = set_tokenizer(category_list, split_char=' ', max_len=max_len)
    # print('序列化完成')
    trian_save_word2vec(category_list, window=int(max_len / 2), save_name=config.model_path + 'w2v_model/{}_256.txt'.format(flag),
                        split_char=' ')

    # 19x256