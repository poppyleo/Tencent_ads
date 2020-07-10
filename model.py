import tensorflow as tf
from tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list, \
    transformer_model_alialili, get_activation, create_attention_mask_from_input_mask  # BERT
from tensorflow.contrib.layers.python.layers import initializers
from tf_utils.crf_utils import rnncell as rnn


# import memory_saving_gradients
# tf.__dict__["gradients"] = memory_saving_gradients.gradients_memory
# 对于CRF这种多优化目标的层，memory_saving_gradients会出bug，注释即可。


class Model:

    def __init__(self, config):
        self.config = config
        self.category = tf.placeholder(tf.float32, [None, 14, 256], name='category')
        self.industry = tf.placeholder(tf.float32, [None, 36, 256], name='industry')
        self.product = tf.placeholder(tf.float32, [None, 48, 256], name='product')
        self.advertiser = tf.placeholder(tf.float32, [None, 48, 256], name='advertiser')
        self.creative = tf.placeholder(tf.float32, [None, 80, 256], name='creative')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')

        self.age = tf.placeholder(tf.int32, [None], name='age')  # 年龄标签
        self.gender = tf.placeholder(tf.int32, [None], name='gender')  # 性别标签
        self.merge_label = tf.placeholder(tf.int32, [None], name='merge_label')  # 20分类标签
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.initializer = initializers.xavier_initializer()

        self.age_num = self.config.age_num
        self.gender_num = self.config.gender_num
        self.merge_num = self.age_num * self.gender_num
        self.model_type = self.config.model_type
        self.len_list = self.config.len_list

        print('Run Model Type:', self.model_type)

        output_layer_list = []
        word_embedding_list = [self.category, self.industry, self.product, self.advertiser, self.creative]
        mask_list = [self.input_mask[:, :14], self.input_mask[:, 14:14 + 36], self.input_mask[:, 50:50 + 48],
                     self.input_mask[:, 98: 98 + 48], self.input_mask[:, 146:146 + 80]]
        # word_embedding_list = word_embedding_list[4:]
        # mask_list = mask_list[4:]
        for length, output_layer in enumerate(word_embedding_list):  # 读取五个embedding

            attention_mask = create_attention_mask_from_input_mask(output_layer, mask_list[length])
            batch_size = output_layer.shape[0]
            seq_len = output_layer.shape[1]
            hidden_size = output_layer.shape[2]
            output_layer = transformer_model_alialili(
                input_tensor=output_layer,
                attention_mask=attention_mask,
                hidden_size=256,
                num_hidden_layers=1,
                num_attention_heads=int(self.config.transformer_size / 64),
                intermediate_size=self.config.transformer_size * 4,
                intermediate_act_fn=get_activation("gelu"),
                hidden_dropout_prob=0.9,
                attention_probs_dropout_prob=0.9,
                initializer_range=0.02,
                do_return_all_layers=False,
                name=str(length)
            )
            if self.model_type == 'bilstm':
                bilstm_inputs = tf.nn.dropout(output_layer, self.config.dropout)
                # bi-directional lstm layer
                bilstm_cell_fw = tf.contrib.rnn.LSTMCell(self.config.lstm_dim, name=str(length) + 'fw')  # 参数可调试
                bilstm_cell_bw = tf.contrib.rnn.LSTMCell(self.config.lstm_dim, name=str(length) + 'bw')  # 参数可调试
                output_layer_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=bilstm_cell_fw,
                                                                 cell_bw=bilstm_cell_bw,
                                                                 inputs=bilstm_inputs,
                                                                 sequence_length=None,
                                                                 dtype=tf.float32)[0]
                model_outputs = tf.concat([output_layer_1[0], output_layer_1[1]], axis=-1)
                hidden_size = model_outputs.shape[-1]
            elif self.model_type == 'lstm':
                print(self.model_type)
                batch_size = output_layer.shape[0]
                lstm_inputs = tf.nn.dropout(output_layer, self.config.dropout)
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.config.lstm_dim, forget_bias=0.8, state_is_tuple=True)
                with tf.name_scope('initial_state'):
                    self.cell_init_state = lstm_cell.zero_state(self.config.batch_size, dtype=tf.float32)
                cell_outputs,_= tf.nn.dynamic_rnn(
                    lstm_cell, lstm_inputs, initial_state=self.cell_init_state, time_major=False)
                model_outputs =cell_outputs
                hidden_size = model_outputs.shape[-1]

            elif self.model_type == 'gru':
                print(self.model_type)
                gru_inputs = tf.nn.dropout(output_layer, self.config.dropout)
                # bi-directional gru layer
                GRU_cell_fw = tf.contrib.rnn.GRUCell(self.config.gru_num, name=str(length) + 'fw')  # 参数可调试
                # 后向
                GRU_cell_bw = tf.contrib.rnn.GRUCell(self.config.gru_num, name=str(length) + 'bw')  # 参数可调试
                output_layer_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                 cell_bw=GRU_cell_bw,
                                                                 inputs=gru_inputs,
                                                                 sequence_length=None,
                                                                 dtype=tf.float32)[0]
                model_outputs = tf.concat([output_layer_1[0], output_layer_1[1]], axis=-1)
                hidden_size = model_outputs.shape[-1]

            else:  # only bert_output
                model_outputs = output_layer
                # pool_size = seq_len
                hidden_size = get_shape_list(output_layer)[-1]

            if self.config.is_avg_pool:
                print('is_avg_pool:', self.config.is_avg_pool)
                output_layer = model_outputs
                # avpooled_out = tf.layers.average_pooling1d(output_layer, pool_size=seq_len, strides=1)
                print(output_layer)
                avpooled_out = tf.layers.max_pooling1d(output_layer, pool_size=self.config.len_list[length], strides=1)
                print(avpooled_out)
                # shape = [batch, hidden_size]
                avpooled_out = tf.reshape(avpooled_out, [-1, hidden_size])
            else:
                print('CLS:', True)
                avpooled_out = output_layer[:, 0:1, :]  # pooled_output
                avpooled_out = tf.squeeze(avpooled_out, axis=1)

            output_layer_list.append(avpooled_out)
            # pool_out_logits = avpooled_out
        pool_out_logits = tf.concat(output_layer_list, axis=-1)
        # pool_out_logits = avpooled_out
        print(pool_out_logits)
        def logits_and_predict(avpooled_out, num_classes, name_scope=None):
            with tf.name_scope(name_scope):
                inputs = tf.nn.dropout(avpooled_out, keep_prob=self.keep_prob)  # delete dropout
                logits = tf.layers.dense(inputs, num_classes,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         name=name_scope + '_logits')
                predict = tf.round(tf.sigmoid(logits), name=name_scope + "predict")
            return logits, predict

        """性别年龄是否用同一条向量"""
        if self.config.joint:
            # 联合学习任务
            avpooled_out_1 = pool_out_logits
            # avpooled_out_2= avpooled_out
            self.age_logits, self.age_predict = logits_and_predict(avpooled_out_1, self.age_num,
                                                                   name_scope='age_relation')
            age_one_hot_labels = tf.one_hot(self.age, depth=self.age_num, dtype=tf.float32)

            self.gender_logits, self.gender_predict = logits_and_predict(avpooled_out_1, self.gender_num,
                                                                         name_scope='gender_relation')
            gender_one_hot_labels = tf.one_hot(self.gender, depth=self.gender_num, dtype=tf.float32)

            age_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=age_one_hot_labels, logits=self.age_logits)
            age_loss = tf.reduce_mean(tf.reduce_sum(age_losses, axis=1))

            gender_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=gender_one_hot_labels,
                                                                    logits=self.gender_logits)
            gender_loss = tf.reduce_mean(tf.reduce_sum(gender_losses, axis=1))
            self.loss = self.config.joint_rate[0] * age_loss + self.config.joint_rate[1] * gender_loss
        else:
            # 20分类
            self.merge_logits, self.merge_predict = logits_and_predict(pool_out_logits, self.merge_num,
                                                                       name_scope='merge_relation')
            merge_one_hot_labels = tf.one_hot(self.merge_label, depth=self.merge_num, dtype=tf.float32)
            merge_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=merge_one_hot_labels,
                                                                   logits=self.merge_logits)
            merge_loss = tf.reduce_mean(tf.reduce_sum(merge_losses, axis=1))
            self.loss = merge_loss
