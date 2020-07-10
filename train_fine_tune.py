import os
import time
import json
import tqdm
from config import Config
from model import *
from utils import DataIterator
from optimization import create_optimizer
import numpy as np
from sklearn.metrics import  accuracy_score
import gensim



gpu_id = 2
"""
[: 180000] batch_size = 1024 max_pooling+lstm 
removed_dict:/home/wangzhili/lei/Tencent/data/save_path/runs_2/1592405433
ori_dict  :  
2卡 lstm+creative_id
0卡 out_put +all_id

"""



os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Fine Tune Learning Rate: ', Config().embed_learning_rate)
print('Batch Size: ', Config().batch_size)
print('Use avg pool', Config().is_avg_pool)
print('loss name:', Config().loss_name)
print('weight:', Config().joint_rate)
print('model_type:', Config().model_type)

print('pretrainning_model:', Config().pretrainning_model)


def softmax(x, axis=1):
    """
    自写函数定义softmax
    :param x:
    :param axis:
    :return:
    """
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    #计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def train(train_iter, test_iter, config):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            model = Model(config)  # config.sequence_length,

            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.decay_step,
                                                       config.decay_rate, staircase=True)

            normal_optimizer = tf.train.AdamOptimizer(learning_rate)

            all_variables = graph.get_collection('trainable_variables')
            bert_var_list = [x for x in all_variables if 'leipengbin' in x.name]
            normal_var_list = [x for x in all_variables if 'leipengbin' not in x.name]
            print('bert train variable num: {}'.format(len(bert_var_list)))
            print('normal train variable num: {}'.format(len(normal_var_list)))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int(train_iter.num_records / config.batch_size * config.train_epoch)

            word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                model.loss, config.embed_learning_rate, num_train_steps=num_batch,
                num_warmup_steps=int(num_batch * 0.05) , use_tpu=False ,  variable_list=bert_var_list
            )

            train_op = tf.group(normal_op, word2vec_op)


            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(config.save_model, "runs_" + str(gpu_id), timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
                json.dump(config.__dict__, file)
            print("Writing to {}\n".format(out_dir))

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)
            if config.continue_training:
                print('recover from: {}'.format(config.checkpoint_path))
                saver.restore(session, config.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            cum_step = 0
            for i in range(config.train_epoch):
                for cate_mat_list, industry_mat_list, product_mat_list, advertiser_mat_list, creative_mat_list,\
               input_masks_list, age_list, gender_list, all_label_list in tqdm.tqdm(
                        train_iter):
                    feed_dict = {
                        model.category: np.array(cate_mat_list),
                        model.industry: np.array(industry_mat_list),
                        model.product: np.array(product_mat_list),
                        model.advertiser: np.array(advertiser_mat_list),
                        model.creative: np.array(creative_mat_list),
                        model.input_mask: np.array(input_masks_list),
                        model.age: age_list,
                        model.gender: gender_list,
                        model.merge_label : all_label_list,
                        model.keep_prob: config.keep_prob,
                    }

                    _, step, _, loss, lr = session.run(
                            fetches=[train_op,
                                     global_step,
                                     embed_step,
                                     model.loss,
                                     learning_rate
                                     ],
                            feed_dict=feed_dict)

                    if cum_step % int(config.decay_step/2)  == 0:
                        format_str = 'step {}, loss {:.4f} lr {:.5f}'
                        print(
                            format_str.format(
                                step, loss, lr)
                        )
                    cum_step += 1

                age_auc, gender_auc = set_test(model, test_iter, session)
                print('dev set : cum_step_{},age_auc_{},gender_auc_{}'.format(cum_step, age_auc, gender_auc))
                if gender_auc>=0.935 or i==2:
                    saver.save(session, os.path.join(out_dir, 'model_{:.4f}_{:.4f}'.format(age_auc, gender_auc)), global_step=step)


def set_test(model, test_iter, session):

    if not test_iter.is_test:
        test_iter.is_test = True

    true_age_list = []
    pred_age_list = []
    true_gender_list = []
    pred_gender_list = []
    for cate_mat_list, industry_mat_list, product_mat_list, advertiser_mat_list, creative_mat_list,\
               input_masks_list, age_list, gender_list, all_label_list in tqdm.tqdm(
            test_iter):

        feed_dict = {
            model.category: np.array(cate_mat_list),
            model.industry: np.array(industry_mat_list),
            model.product: np.array(product_mat_list),
            model.advertiser: np.array(advertiser_mat_list),
            model.creative: np.array(creative_mat_list),
            model.input_mask: np.array(input_masks_list),
            model.age: age_list,
            model.gender: gender_list,
            model.merge_label: all_label_list,
            model.keep_prob: config.keep_prob,
        }

        age_label, gender_label = session.run(
            fetches=[model.age_logits, model.gender_logits],
            feed_dict=feed_dict
        )

        age_label=softmax(age_label)
        gender_label=softmax(gender_label)

        age_label = np.argmax(age_label, axis=1)
        gender_label = np.argmax(gender_label, axis=1)

        true_age_list.extend(age_list)
        true_gender_list.extend(gender_list)

        pred_age_list.extend(age_label)
        pred_gender_list.extend(gender_label)

    assert len(true_age_list)==len(pred_gender_list)

    age_auc = accuracy_score(true_age_list,pred_age_list)
    gender_auc = accuracy_score(true_gender_list,pred_gender_list)
    print('focal_auc {}, age_auc {}, gender_auc {}'.format(age_auc+gender_auc, age_auc,gender_auc))

    return age_auc, gender_auc


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

    embeddings_5 = gensim.models.KeyedVectors.load_word2vec_format(config.model_path + 'w2v_model/creative_256.txt',
                                                                   binary=False)
    print('model_loadding done')
    train_iter = DataIterator(config.batch_size, embeddings_1, embeddings_2, embeddings_3, embeddings_4, embeddings_5,
                             data_file=config.corpus_path + 'new_train.csv', seq_length=config.sequence_length,config=config)

    dev_iter =  DataIterator(config.batch_size, embeddings_1, embeddings_2, embeddings_3, embeddings_4, embeddings_5,
                             data_file=config.corpus_path + 'new_dev.csv', seq_length=config.sequence_length,config=config)

    train(train_iter, dev_iter, config)
