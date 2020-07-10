import pandas as pd
from config import Config

config = Config()

# def sort_click_times(x):
#     x=x.replace(' 天 ','/')
#     x=x.replace(' 地 ','/')
#     x=x.replace(' 玄 ','/')
#     x=x.replace(' 黄 ','/')
#     x=x.replace(';','/')
#     text = x.split(';')[0]
#     click_times = x.split(';')[0]


if __name__ == '__main__':
    train_df = pd.read_csv(config.corpus_path + 'train.csv')

    # 切分训练集，分成训练集和验证集
    print('Train Set Size:', train_df.shape)
    new_dev_df = train_df[: 180000]
    frames = [train_df[180000:360000], train_df[360000:]]
    new_train_df = pd.concat(frames)  # 训练集

    new_train_df = new_train_df.fillna('')

    new_train_df.to_csv(config.corpus_path + 'new_train.csv', index=False)
    new_dev_df.to_csv(config.corpus_path + 'new_dev.csv', index=False)
