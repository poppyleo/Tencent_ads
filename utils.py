import numpy as np
from tqdm import tqdm
from config import Config
import pandas as pd
import os
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

gpu_id = 5
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


# 得到embedding矩阵
def findmatriwithpadding(embedding, word_list, max_len):
    """

    :param embedding:word_embedding模型
    :param word_list: 字表
    :param max_len: 256
    :return:
    """
    embedding_matrix = np.zeros((max_len, 256))
    for i in range(max_len):
        try:
            word = word_list[i]
            cur_vector = embedding[word]
        except:
            """word长度不够，padding 0 矩阵"""
            cur_vector = np.zeros(256)
        embedding_matrix[i] = cur_vector
    return embedding_matrix


### Tokenizer 序列化文本
def set_tokenizer(docs, split_char=' ', max_len=100):
    '''
    输入
    docs:文本列表
    split_char:按什么字符切割
    max_len:截取的最大长度

    输出
    X:序列化后的数据
    word_index:文本和数字对应的索引
    '''
    tokenizer = Tokenizer(lower=False, char_level=False, split=split_char)
    tokenizer.fit_on_texts(docs)
    X = tokenizer.texts_to_sequences(docs)
    maxlen = max_len
    X = pad_sequences(X, maxlen=maxlen, value=0)
    word_index = tokenizer.word_index
    return X, word_index


def load_data(data_file):
    """
    读取数据
    :param file:
    :return:
    """
    data_df = pd.read_csv(data_file)
    data_df.fillna('', inplace=True)
    lines = list(zip(list(data_df['text']), list(data_df['age']), list(data_df['gender'])))

    return lines


def pad(x, max_len):
    x = list(x)
    while len(x) < max_len:
        x.append(0)
    if len(x) > max_len:
        x = x[:max_len]
    return x


def create_example(lines):
    examples = []
    for (_i, line) in enumerate(lines):
        text = str(line[0])
        age = int(line[1])
        gender = int(line[2])

        examples.append(InputExample(text=text, age=age, gender=gender))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text, age, gender):
        self.text = text
        self.age = age
        self.gender = gender
        self.all_label = str(gender) + str(age)


class DataIterator:
    """
    数据迭代器
    """

    def __init__(self, batch_size, embedding_1, embedding_2, embedding_3, embedding_4, embedding_5, data_file,
                 seq_length=100, is_test=False, config=None):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        self.embeddings_1 = embedding_1
        self.embeddings_2 = embedding_2
        self.embeddings_3 = embedding_3
        self.embeddings_4 = embedding_4
        self.embeddings_5 = embedding_5
        self.config = config

        if not self.is_test:
            self.shuffle()
        print(self.num_records)

    def convert_single_example(self, example_idx):
        text = self.data[example_idx].text
        age = self.data[example_idx].age - 1
        gender = self.data[example_idx].gender - 1
        all_label = self.data[example_idx].all_label
        self.config.save_dict['-1-1']=100
        all_label = self.config.save_dict[all_label]
        x = text.replace(' 天 ', '/')
        x = x.replace(' 地 ', '/')
        x = x.replace(' 玄 ', '/')
        x = x.replace(' 黄 ', '/')
        x = x.replace(';', '/')
        category_id = x.split('/')[0].split(' ')
        industry_id = x.split('/')[1].split(' ')
        product_id = x.split('/')[2].split(' ')
        advertiser_id = x.split('/')[3].split(' ')
        creative_id = x.split('/')[4].split(' ')
        click_times = [int(i) for i in x.split('/')[5].split(' ')]

        index = len(category_id) + 1
        category_times = click_times[:index]
        industry_times = click_times[index:index + len(industry_id) + 1]
        index += len(industry_id) + 1
        product_times = click_times[index:index + len(product_id) + 1]
        index += len(product_id) + 1
        advertiser_times = click_times[index:index + len(advertiser_id) + 1]
        index += len(advertiser_id) + 1
        creative_times = click_times[index:index + len(creative_id) + 1]

        category_id, category_times = zip(*sorted(zip(category_id, category_times), key=lambda x: x[1], reverse=True))
        industry_id, industry_times = zip(*sorted(zip(industry_id, industry_times), key=lambda x: x[1], reverse=True))
        product_id, product_times = zip(*sorted(zip(product_id, product_times), key=lambda x: x[1], reverse=True))
        advertiser_id, advertiser_times = zip(*sorted(zip(advertiser_id, advertiser_times), key=lambda x: x[1], reverse=True))
        creative_id, creative_times = zip(*sorted(zip(creative_id, creative_times), key=lambda x: x[1], reverse=True))

        input_masks = pad(category_times, 14) + pad(industry_times, 36) + pad(product_times, 48) + \
                      pad(advertiser_times, 48) + pad(creative_times, 80)
        cate_mat = findmatriwithpadding(self.embeddings_1, category_id, 14)
        industry_mat = findmatriwithpadding(self.embeddings_2, industry_id, 36)
        product_mat = findmatriwithpadding(self.embeddings_3, product_id, 48)
        advertiser_mat = findmatriwithpadding(self.embeddings_4, advertiser_id, 48)
        creative_mat = findmatriwithpadding(self.embeddings_5, creative_id, 80)
        # input_masks = [1]*(14 + 36 + 48 + 48 + 80)
        assert len(cate_mat) + len(industry_mat) + len(product_mat) + len(advertiser_mat) + len(creative_mat) == len(
            input_masks)
        assert len(input_masks) == 14 + 36 + 48 + 48 + 80
        return cate_mat, industry_mat, product_mat, advertiser_mat, creative_mat, input_masks, age, gender, all_label

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        cate_mat_list = []
        industry_mat_list = []
        product_mat_list = []
        advertiser_mat_list = []
        creative_mat_list = []
        input_masks_list = []
        age_list = []
        gender_list = []
        all_label_list = []
        num_tags = 0

        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            cate_mat, industry_mat, product_mat, advertiser_mat, creative_mat, input_masks, age, gender, all_label = res

            # 一个batch的输入
            cate_mat_list.append(cate_mat)
            industry_mat_list.append(industry_mat)
            product_mat_list.append(product_mat)
            advertiser_mat_list.append(advertiser_mat)
            creative_mat_list.append(creative_mat)
            input_masks_list.append(input_masks)
            age_list.append(age)
            gender_list.append(gender)
            all_label_list.append(all_label)
            num_tags += 1
            self.idx += 1
            if self.idx >= self.num_records:
                break

        return cate_mat_list, industry_mat_list, product_mat_list, advertiser_mat_list, creative_mat_list, \
               input_masks_list, age_list, gender_list, all_label_list


if __name__ == '__main__':
    config = Config()
    # 得到emb矩阵
    print('loading word2vec mat1...')
    embeddings_1 = gensim.models.KeyedVectors.load_word2vec_format(config.model_path + 'w2v_model/category_256.txt',
                                                                   binary=False)
    print('loading word2vec mat2...')
    embeddings_2 = gensim.models.KeyedVectors.load_word2vec_format(config.model_path + 'w2v_model/industry_256.txt',
                                                                   binary=False)
    print('loading word2vec mat3...')
    embeddings_3 = gensim.models.KeyedVectors.load_word2vec_format(config.model_path + 'w2v_model/product_256.txt',
                                                                   binary=False)
    print('loading word2vec mat4...')
    embeddings_4 = gensim.models.KeyedVectors.load_word2vec_format(config.model_path + 'w2v_model/advertiser_256.txt',
                                                                   binary=False)
    print('loading word2vec mat5...')
    print('model_loadding done')
    embeddings_5 = gensim.models.KeyedVectors.load_word2vec_format(config.model_path + 'w2v_model/creative_256.txt',
                                                                   binary=False)

    data_iter = DataIterator(config.batch_size, embeddings_1, embeddings_2, embeddings_3, embeddings_4, embeddings_5,
                             data_file=config.corpus_path + 'train.csv', seq_length=config.sequence_length,
                             config=config)
    #

    for cate_mat_list, industry_mat_list, product_mat_list, advertiser_mat_list, creative_mat_list, \
        input_masks_list, age_list, gender_list, all_label_list in tqdm(
        data_iter):
        print(cate_mat_list[0].shape)
        break
