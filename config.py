class Config:

    def __init__(self):
        # self.data_path = '/home/wangzhili/CY_Autumn/Tencent_AD/Data/'
        # self.data_processed = '/home/wangzhili/lei/Tencent/data/'
        # self.train_type = 'creative_id' #''ad':ad_id+others 'creative_id+others'
        # self.finetune_data = '/home/wangzhili/lei/Tencent/finetune_data/'

        # 追一
        """新策略路径"""
        # self.data_path = '/home/none404/hm/Tencent_ads/data/'
        # self.data_processed = '/home/none404/hm/Tencent_ads/data/processed_xuan/'
        # self.train_type = 'creative' #''ad':ad_id+others 'creative_id+others'
        # self.finetune_data = '/home/none404/hm/Tencent_ads/finetune_xuan/'
        # self.finetune_xuan = '/home/none404/hm/Tencent_ads/finetune_xuan/finetune_data/'
        # self.data4transformer = '/home/none404/hm/Tencent_ads/data/one_transformer/'

        # 李飞鹏
        # self.merge_path = '/mnt/inspurfs/user-fs/lifeipeng/data/corpus/128_4pid_4aid/'
        # self.ori_path = '/mnt/inspurfs/user-fs/lifeipeng/Autumn_CY/Tencent_AD/Data/'
        # self.corpus_path = '/mnt/inspurfs/user-fs/lifeipeng/data/corpus/128_4pid_4aid/Ori_creative_data/'

        # # wzl
        self.embed_dense = True
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.over_sample = True

        self.num_checkpoints = 20 * 3

        self.train_epoch = 100
        self.sequence_length = 256

        self.learning_rate = 1e-4 * 5
        self.embed_learning_rate = 3e-5 *5

        self.batch_size = 1024
        self.test_batch_size = 256  # 测试集batch_size
        self.decay_rate = 0.5
        self.decay_step = int(720000 / self.batch_size)
        self.embed_trainable = True

        self.as_encoder = True
        self.age_num = 10
        self.gender_num = 2
        self.save_model = '/home/wangzhili/lei/Tencent/data/save_path/'
        # self.merge_path = '/home/wangzhili/lei/Tencent/data/data/'
        # self.ori_path = '/home/wangzhili/lei/Tencent/data/data/'
        # self.corpus_path = '/home/wangzhili/lei/Tencent/data/deal_data/'
        self.continue_training = False
        self.pretrainning_model='word2vec'
        self.data_path = '/home/wangzhili/CY_Autumn/Tencent_AD/Data/'

        self.corpus_path = '/home/wangzhili/lei/Tencent/data/datanot_delet/'
        self.model_path = '/home/wangzhili/lei/Tencent/data/alldict_model_path/'  # 原始数据
        #
        self.result = '/home/wangzhili/lei/Tencent/data/result/'
        self.model_path = '/home/wangzhili/lei/Tencent/data/model_path/'#去除桶子
        self.corpus_path = '/home/wangzhili/lei/Tencent/data/deal_data/'

        self.checkpoint_path = '/home/wangzhili/lei/Tencent/data/save_path/runs_0/1592532385/model_0.1239_0.9307-2112'
        self.checkpoint_path = "/home/wangzhili/lei/Tencent/data/save_path/runs_0/1592561094/model_0.1650_0.9353-22528"

        self.sequence_length = 16 + 36 + 48 + 48 + 80
        self.age_num = 10
        self.gender_num = 2
        self.is_avg_pool = True  # True: 使用平均avg_pool False:使用CLS
        self.joint = True  # True联合学习任务 False:20分类
        self.joint_rate = [0, 1]  # 联合学习loss权值

        self.model_type = 'bilstm'
        # self.model_type = 'gru'
        # self.model_type = 'lstm'

        # self.model_type = 'only bert output'
        self.lstm_dim = 256
        self.dropout = 0.9
        self.learning_rate = 1e-4
        self.embed_learning_rate = 3e-5
        # self.loss_name = 'focal_loss'
        self.loss_name = 'normal'
        self.gru_num = 256
        self.keep_prob = 0.9
        self.transformer_size = 256
        self.len_list = [14, 36, 48, 48, 80]
        self.save_dict = {'11': 0, '12': 1, '13': 2, '14': 3, '15': 4, '16': 5, '17': 6, '18': 7, '19': 8, '110': 9,
                          '21': 10, '22': 11, '23': 12, '24': 13, '25': 14, '26': 15, '27': 16, '28': 17, '29': 18,
                          '210': 19
                          }  # 保存的字典  第一位为性别
