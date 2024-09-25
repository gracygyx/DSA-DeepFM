import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
import time
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.python import debug as tf_debug
import pandas as pd
import networkx as nx
import joblib
import math
import os
import numpy as np
from tensorflow.python.framework import ops
import subprocess
import shutil

class DeepFM(BaseEstimator, TransformerMixin):
    def __init__(self,
                 cate_feature_size,
                 numeric_feature_size,
                 field_size,
                 embedding_size=30,

                 dropout_fm=[1.0, 1.0],
                 deep_layers_fm=[360, 720],

                 dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,

                 initial_rate=0.001,
                 decay_rate=0.9,

                 epoch=2,
                 batch_size=64,
                 optimizer_type="adam",

                 verbose=False,
                 random_seed=2023,

                 use_fm=True,
                 use_deep=True,
                 use_attention=True,
                 use_bn=True,
                 batch_norm_decay=0.995,

                 loss_type="logloss",
                 eval_metric=roc_auc_score,
                 l2_reg=0.0,
                 path='./',
                 greater_is_better=True):
        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"  # 'logloss' for classification

        self.cate_feature_size = cate_feature_size  # 1506 Size of the feature dictionary：1506
        self.numeric_feature_size = numeric_feature_size  # 7460

        self.field_size = field_size  # Size of the feature fields：3
        self.embedding_size = embedding_size  # Size of the feature embedding: 30
        self.total_size_FM = 2 * self.field_size * self.embedding_size  # 90

        self.dropout_fm = dropout_fm  # [1, 1]
        self.deep_layers_fm = deep_layers_fm  # [360, 720]
        self.deep_layers_num = [2048]
        self.dropout_deep = dropout_deep  # [0.5, 0.5, 0.5]
        self.deep_layers_activation = deep_layers_activation  # Relu

        self.initial_rate = initial_rate
        self.decay_rate = decay_rate

        self.use_fm = use_fm
        self.use_deep = use_deep
        self.use_attention = use_attention

        self.use_bn = use_bn
        self.batch_norm_decay = batch_norm_decay

        self.l2_reg = l2_reg
        self.epoch = epoch  # 128
        self.batch_size = batch_size  # 64
        self.optimizer_type = optimizer_type  # 'adam'

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type  # 'logloss'
        self.eval_metric = eval_metric  # roc_auc_score
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []
        self.attention_output_dim = 2  # 90
        self.attention_fm_num = Attention(self.embedding_size, self.attention_output_dim)
        self.attention_fm_num_transpose = Attention(self.field_size, self.attention_output_dim)
        self.attention_res = Attention(self.embedding_size, self.attention_output_dim)
        self.attention_res_transpose = Attention(self.field_size, self.attention_output_dim)

        self.path = path

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_epoch = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(learning_rate=self.initial_rate, global_step=self.global_epoch,
                                                            decay_rate=self.decay_rate,
                                                            decay_steps=1, staircase=True)  # 指数衰减学习率
            self._init_graph()

        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for var in tf.global_variables():
            shape = var.shape  # 获取每个变量的shape，其类型为'tensorflow.python.framework.tensor_shape.TensorShape'
            array = np.asarray([dim.value for dim in shape])  # 转换为numpy数组，方便后续计算
            mulValue = np.prod(array)  # 使用numpy prod接口计算数组所有元素之积

            Total_params += mulValue  # 总参数量
            if var.trainable:
                Trainable_params += mulValue  # 可训练参数量
                print(var)
            else:
                NonTrainable_params += mulValue  # 非可训练参数量

    def _init_graph(self):
        with self.graph.as_default():
            # init
            tf.set_random_seed(self.random_seed)

            self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name="feat_index")
            self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name="feat_value")
            self.numeric_value = tf.placeholder(tf.float32, [None, None], name='num_value_drug_drug_cell')
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="label")

            self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
            self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
            self.train_phase = tf.placeholder(tf.bool, name="train_phase")

            self.weights = self._initialize_weights()

            # model get the embedding weight
            self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_index)  # [batch, 3, 30]
            # get feature value
            feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1])  # [batch, 3, 30]
            # get the embeding of each node
            self.embeddings = tf.multiply(self.embeddings, feat_value)  # [batch, 3, 30]

            if self.use_bn:
                self.embeddings = self.batch_norm_layer(self.embeddings, train_phase=self.train_phase, scope_bn='embeddings')


            # get numerical feature value
            # num_layer_0
            self.numeric_value_0 = tf.add(tf.matmul(self.numeric_value, self.weights["feature_numerical_0"]),
                                          self.weights["bias_feature_numerical_0"])  # [batch, 2048]
            if self.use_bn:
                self.numeric_value_0 = self.batch_norm_layer(self.numeric_value_0, train_phase=self.train_phase,
                                                             scope_bn='numeric_value_0')  # [batch, 2048]
            self.numeric_value_0 = self.deep_layers_activation(self.numeric_value_0)  # [batch, 2048]
            self.numeric_value_0 = tf.nn.dropout(self.numeric_value_0, self.dropout_keep_deep[0])  # [batch, 2048]

            # num_layer_1
            self.numeric_value_0 = tf.add(tf.matmul(self.numeric_value_0, self.weights["feature_numerical_1"]),
                                          self.weights["bias_feature_numerical_1"])  # [batch, 1536]
            if self.use_bn:
                self.numeric_value_0 = self.batch_norm_layer(self.numeric_value_0, train_phase=self.train_phase,
                                                             scope_bn='numeric_value_1')  # [batch, 1536]
            self.numeric_value_0 = self.deep_layers_activation(self.numeric_value_0)  # [batch, 1536]
            self.numeric_value_0 = tf.nn.dropout(self.numeric_value_0, self.dropout_keep_deep[1])  # [batch, 1536]

            # attention
            if self.use_attention:
                x_k = self.embeddings    # [batch, 90]
                x_v = tf.reshape(self.numeric_value_0, (-1, self.field_size, self.embedding_size))    # [batch, 90]

                self.embeddings, self.numeric_value_0 = self.attention_fm_num([x_k, x_v])  # [batch, 90]
                self.embeddings, self.numeric_value_0 = self.attention_fm_num_transpose(
                    [tf.transpose(self.embeddings, [0, 2, 1]), tf.transpose(self.numeric_value_0, [0, 2, 1])])
                self.numeric_value_0 = tf.transpose(self.numeric_value_0, [0, 2, 1])
                self.numeric_value_0 = tf.reshape(self.numeric_value_0,
                                                  shape=[-1, self.field_size * self.embedding_size])
                self.embeddings = tf.transpose(self.embeddings, [0, 2, 1])
                self.embeddings = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
                if self.use_bn:
                    self.embeddings = self.batch_norm_layer(self.embeddings, train_phase=self.train_phase,
                                                            scope_bn='embedding_att')
                    self.numeric_value_0 = self.batch_norm_layer(self.numeric_value_0, train_phase=self.train_phase,
                                                                 scope_bn='num_att')
                self.x0 = tf.concat([self.embeddings, self.numeric_value_0], axis=1)
            else:
                self.embeddings = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])
                if self.use_bn:
                    self.embeddings = self.batch_norm_layer(self.embeddings, train_phase=self.train_phase,
                                                            scope_bn='embedding_att')
                    self.numeric_value_0 = self.batch_norm_layer(self.numeric_value_0, train_phase=self.train_phase,
                                                                 scope_bn='num_att')
                self.x0 = tf.concat([self.embeddings, self.numeric_value_0], axis=1)

            # ---------- first order term ----------
            self.y_first_order = tf.matmul(self.x0, self.weights['feature_bias'])  # [batch, 360]

            # ---------- second order term ---------------
            # sum_square part
            self.summed_features_emb = tf.matmul(self.x0, self.weights['feature_vector'])  # [batch, 360]
            self.summed_features_emb_square = tf.square(self.summed_features_emb)   # [batch, 360]

            # square_sum part
            self.squared_sum_features_emb = tf.matmul(tf.square(self.x0), tf.square(self.weights['feature_vector']))  # [batch, 360]

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # [batch, 360]
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # [batch, 360]

            # ---------- Deep component ----------
            self.y_deep = tf.reshape(self.x0, shape=[-1, self.total_size_FM])  # [batch, 90]
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])  # [batch, 90]

            for i in range(0, len(self.deep_layers_fm )):  # i=0, i=1
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i])  # [batch, 180]; [batch, 360]
                if self.use_bn:
                    self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn='y_deep_'+str(i))  # [batch, 180]; [batch, 360]

                self.y_deep = self.deep_layers_activation(self.y_deep)  # [batch, 180]; [batch, 360]
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])  # [batch, 180]; [batch, 360]


            # ---------- DeepFM ----------
            if self.use_fm and self.use_deep:
                concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)  # [batch, 2048]
            elif self.use_fm:
                concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
            elif self.use_deep:
                concat_input = self.y_deep

            self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection_0"]), self.weights["concat_bias_0"])  # [batch, 1536]
            if self.use_bn:
                self.out = self.batch_norm_layer(self.out, train_phase=self.train_phase, scope_bn='out_0')  # [batch, 1536]
            self.out = self.deep_layers_activation(self.out)  # [batch, 90]
            self.out = tf.nn.dropout(self.out, self.dropout_keep_deep[0])

            # attention
            if self.use_attention:
                x_k = tf.reshape(self.x0, (-1, 2 * self.field_size, self.embedding_size))  # [batch, 90]
                x_v = tf.reshape(self.out, (-1, 2 * self.field_size, self.embedding_size))    # [batch, 90]

                self.x0, self.out = self.attention_res([x_k, x_v])  # [batch, 90]
                self.x0, self.out = self.attention_res_transpose(
                    [tf.transpose(self.x0, [0, 2, 1]), tf.transpose(self.out, [0, 2, 1])])
                self.x0 = tf.transpose(self.x0, [0, 2, 1])
                self.x0 = tf.reshape(self.x0, shape=[-1, 2 * self.field_size * self.embedding_size])
                self.out = tf.transpose(self.out, [0, 2, 1])
                self.out = tf.reshape(self.out, shape=[-1, 2 * self.field_size * self.embedding_size])
                if self.use_bn:
                    self.x0 = self.batch_norm_layer(self.x0, train_phase=self.train_phase, scope_bn='res_x0_att')
                    self.out = self.batch_norm_layer(self.out, train_phase=self.train_phase, scope_bn='res_out_att')
                self.out = tf.concat([self.out, self.x0], axis=1)
            else:
                self.out = tf.concat([self.out, self.x0], axis=1)

            self.out = tf.add(tf.matmul(self.out, self.weights["concat_projection_1"]), self.weights["concat_bias_1"])  # [batch, 512]
            if self.use_bn:
                self.out = self.batch_norm_layer(self.out, train_phase=self.train_phase, scope_bn='out_1')  # [batch, 512]
            self.out = self.deep_layers_activation(self.out)  # [batch, 90]
            self.out = tf.nn.dropout(self.out, self.dropout_keep_deep[0])

            # self.reconstruct = tf.matmul(self.out, tf.transpose(self.out)) / (
            #             tf.norm(self.out) * tf.norm(tf.transpose(self.out)))
            # self.original = tf.matmul(self.numeric_value, tf.transpose(self.numeric_value)) / (
            #             tf.norm(self.numeric_value) * tf.norm(tf.transpose(self.numeric_value)))

            self.out = tf.add(tf.matmul(self.out, self.weights["concat_projection_2"]),
                              self.weights["concat_bias_2"])  # [batch, 1]


            # loss
            if self.loss_type == "logloss":
                self.out = tf.nn.sigmoid(self.out)
                self.loss = tf.losses.log_loss(self.label, self.out)
            elif self.loss_type == "mse":
                self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))

            # l2 regularization on weights
            if self.l2_reg > 0:
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection_0"])
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection_1"])
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection_2"])
                if self.use_deep:
                    for i in range(len(self.deep_layers_fm)):  # i=0, 1
                        self.loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d" % i])


            # optimizer
            update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # optimizer
                if self.optimizer_type == "adam":
                    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                            epsilon=1e-8).minimize(self.loss, global_step=self.global_epoch)
                elif self.optimizer_type == "adagrad":
                    self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                               initial_accumulator_value=1e-8).minimize(self.loss)
                elif self.optimizer_type == "gd":
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                elif self.optimizer_type == "momentum":
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                        self.loss)
                elif self.optimizer_type == "yellowfin":
                    self.optimizer = YFOptimizer(learning_rate=self.learning_rate, momentum=0.0).minimize(
                        self.loss)


            # init
            self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
            self.sess = self._init_session()

            # Check if there is a checkpoint
            ckpt = tf.train.get_checkpoint_state(self.path + '/checkpoint')

            # if ckpt and ckpt.model_checkpoint_path:
            if ckpt and ckpt.model_checkpoint_path:
                print(f"Restoring model from {ckpt.model_checkpoint_path}")
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)

                # Extract the epoch number from the checkpoint path
                epoch_number_check = int(ckpt.model_checkpoint_path.split('-')[-1]) + 1
                self.sess.run(tf.assign(self.global_epoch, epoch_number_check))
            else:
                print("Initializing model from scratch")
                init = tf.global_variables_initializer()
                self.sess.run(init)


            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _init_session(self):
        config = tf.ConfigProto(device_count={"gpu": 0})
        config.gpu_options.allow_growth = True
        return tf.Session(config=config)


    def _initialize_weights(self):  # initizing all the weights parameters
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.cate_feature_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # [1506, 30]

        # num
        glorot = np.sqrt(2.0 / (self.numeric_feature_size + self.deep_layers_num[0]))
        weights["feature_numerical_0"] = tf.Variable(
            tf.random_uniform([self.numeric_feature_size, self.deep_layers_num[0]], 0.0, 0.01),
            name="feature_numerical_0")  # [7460, 2048]
        weights["bias_feature_numerical_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_num[0])), dtype=np.float32)  # [1, 2048]

        glorot = np.sqrt(2.0 / (self.deep_layers_num[0] + self.embedding_size * self.field_size))
        weights["feature_numerical_1"] = tf.Variable(
            tf.random_uniform([self.deep_layers_num[0], self.embedding_size * self.field_size], 0.0, 0.01),
            name="feature_numerical_1")  # [512, 90]
        weights["bias_feature_numerical_1"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.embedding_size * self.field_size)), dtype=np.float32)  # [1, 90]

        # DeepFM
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.total_size_FM, self.deep_layers_fm[1]], 0.0, 1.0),
            name="feature_bias")  # [1536, 360]

        weights["feature_vector"] = tf.Variable(
            tf.random_uniform([self.total_size_FM, self.deep_layers_fm[1]], 0.0, 0.01),
            name="feature_vector")  # [90, 360]

        # deep layers
        num_layer = len(self.deep_layers_fm )  # num_layer = 2
        input_size = self.total_size_FM  # input_size = 180
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers_fm[0]))

        weights["layer_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers_fm[0])),
                                         dtype=np.float32)  # [180, 360]

        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_fm[0])),
                                        dtype=np.float32)  # [1, 360]
        for i in range(1, num_layer):  # i = 1
            glorot = np.sqrt(2.0 / (self.deep_layers_fm[i - 1] + self.deep_layers_fm[i]))

            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers_fm[i - 1], self.deep_layers_fm[i])),
                dtype=np.float32)  # [360, 720]

            weights["bias_%d" % i] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers_fm[i])),
                                                 dtype=np.float32)  # [1, 720]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = 3 * self.deep_layers_fm[-1]  # 1440
        elif self.use_fm:
            input_size = 2 * self.deep_layers_fm[0]
        elif self.use_deep:
            input_size = self.deep_layers_fm[-1]

        glorot = np.sqrt(2.0 / (input_size + self.total_size_FM))
        weights["concat_projection_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.total_size_FM)),
            dtype=np.float32)  # [1440, 90]
        weights["concat_bias_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.total_size_FM)), dtype=np.float32)  # [1, 90]

        glorot = np.sqrt(2.0 / (2 * self.total_size_FM + self.embedding_size))
        weights["concat_projection_1"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot,
                             size=(2 * self.total_size_FM, self.embedding_size)),
            dtype=np.float32)  # [90, 90]
        weights["concat_bias_1"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(1, self.embedding_size)), dtype=np.float32)  # [1, 90]

        glorot = np.sqrt(2.0 / (self.embedding_size + 1))
        weights["concat_projection_2"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(self.embedding_size, 1)), dtype=np.float32)  # [90, 1]
        weights["concat_bias_2"] = tf.Variable(tf.constant(0.01), dtype=np.float32)  # []

        return weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                              updates_collections=ops.GraphKeys.UPDATE_OPS,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True,
                                  updates_collections=ops.GraphKeys.UPDATE_OPS,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def get_batch(self, indices_all, batch_size, index):
        start = index * batch_size
        end = (index + 1) * batch_size
        end = end if end < indices_all.shape[0] else indices_all.shape[0]
        return indices_all[start:end]

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a):
        rng_state = np.random.get_state()
        np.random.set_state(rng_state)
        np.random.shuffle(a)

    def fit_on_batch(self, Xi, Xv, Nv, y):
        feed_dict = {self.feat_index: Xi,  # 1024
                     self.feat_value: Xv,
                     self.numeric_value: Nv,
                     self.label: y,  # 1024
                     self.dropout_keep_fm: self.dropout_fm,  # [1, 1]
                     self.dropout_keep_deep: self.dropout_deep,  # [0.5, 0.5, 0.5]
                     self.train_phase: True}
        # loss,loss_l, loss_m, loss_d, opt = self.sess.run((self.loss,self.l_loss, self.m_loss, self.d_loss, self.optimizer), feed_dict=feed_dict)
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)

        # return loss, loss_l, loss_m, loss_d
        return loss

    def fit(self, Xi_train, Xv_train, Nv_train, y_train, Xi_valid=None, Xv_valid=None, Nv_valid=None, y_valid=None,
            early_stopping=False, refit=False):
        has_valid = Xv_valid is not None
        AUC = []
        for epoch in range(self.global_epoch.eval(session=self.sess), self.epoch):
            loss_list = []
            AUC = []
            indices_all = np.arange(y_train.shape[0])
            self.shuffle_in_unison_scary(indices_all)
            total_batch = int(indices_all.shape[0] / self.batch_size)  # total_batch = 65

            for i in range(total_batch):
                start_time = time.time()
                indices_selected = self.get_batch(indices_all, self.batch_size, i)  # len(Xi_batch)=1024
                Xi_batch = Xi_train[indices_selected, :]
                Xv_batch = Xv_train[indices_selected, :]
                Nv_batch = Nv_train[indices_selected, :]
                y_batch = np.array([[y_] for y_ in y_train[indices_selected]])

                self.sess.run(tf.assign(self.global_epoch, epoch))
                lr_value = self.sess.run(self.learning_rate)
                loss = self.fit_on_batch(Xi_batch, Xv_batch, Nv_batch, y_batch)

                end_time = time.time()
                epoch_time = end_time - start_time

                print('Training: Epoch ' + str(epoch+1) + '/' + str(self.epoch) + '; Batch: ' + str(i + 1) + '/' + str(
                    total_batch) + '; Loss: ' + str(loss) + '; Learning rate: ' + str(lr_value) + '; Time: ' + str(epoch_time))
                loss_list.append(loss)

            # Save model checkpoint after completing each epoch
            checkdir = self.path + '/checkpoint/'
            if not os.path.exists(checkdir):
                os.makedirs(checkdir)
            if os.path.exists(checkdir):
                for filename in os.listdir(checkdir):
                    file_path = checkdir + filename
                    if os.path.isfile(file_path) and filename != 'checkpoint':
                        os.remove(file_path)

            self.saver.save(self.sess, self.path + '/checkpoint/DeepFM', global_step=epoch)

            # Write loss to a txt file
            with open(self.path + "loss_epoch.txt", "a") as f:
                for loss_value in loss_list:
                    f.write(str(loss_value) + "\n")

            # Update the global epoch variable after completing an epoch
            self.sess.run(tf.assign(self.global_epoch, epoch + 1))

            # evaluate training and validation datasets
            # train_result_auc = self.evaluate(Xi_train, Xv_train, Nv_train, ggi_c_train, ggi_d1_train,
            #                                              ggi_d2_train, y_train)
            # self.train_result.append(train_result_auc)
            # print('Epoch: ' + str(epoch + 1) + '/' + str(self.epoch), 'Train AUC: ' + str(train_result_auc))

            if has_valid:
                valid_result_auc = self.evaluate(Xi_valid, Xv_valid, Nv_valid, y_valid)
                print('Epoch: ' + str(epoch + 1) + '/' + str(self.epoch), 'Valid AUC: ' + str(valid_result_auc))
                self.valid_result.append(valid_result_auc)
                AUC.append(valid_result_auc)

                if epoch == 0:
                    path_best = self.path + '/Best_model/'
                    path_model = self.path + '/checkpoint/'
                    for filename in os.listdir(path_model):
                        src = os.path.join(path_model, filename)
                        dst = os.path.join(path_best, filename)
                        shutil.copy2(src, dst)
                else:
                    auc_file_all = self.path + 'AUC_valid_epoch.txt'
                    if os.path.exists(auc_file_all):
                        with open(auc_file_all, 'r') as f:
                            auc_values_all = [float(line.strip()) for line in f.readlines()]
                        if all(float(auc) <= AUC[0] for auc in auc_values_all):
                            path_best = self.path + '/Best_model/'
                            path_model = self.path + '/checkpoint/'
                            for filename in os.listdir(path_best):
                                file_path_remove = path_best + filename
                                os.remove(file_path_remove)
                            for filename in os.listdir(path_model):
                                src = os.path.join(path_model, filename)
                                dst = os.path.join(path_best, filename)
                                shutil.copy2(src, dst)

                # Write valid AUC to a txt file
                with open(self.path + "AUC_valid_epoch.txt", "a") as f:
                    for auc_value in AUC:
                        f.write(str(auc_value) + "\n")
            # if self.verbose > 0 and epoch % self.verbose == 0:
            #     if has_valid:
            #         print("[%d] train-result=%.4f, valid-result=%.4f, valid-AUPR=%.4f, [%.1f s]"
            #               % (epoch + 1, train_result_auc, valid_result_auc, valid_AUPR, time() - t1))
            #
            #     else:
            #         print("[%d] train-result=%.4f [%.1f s]"
            #               % (epoch + 1, train_result_auc, time() - t1))
            if has_valid and early_stopping and self.training_termination(self.valid_result):
                # Write valid AUC to a txt file
                with open(self.path + "AUC_valid_epoch.txt", "a") as f:
                    for auc_value in AUC:
                        f.write(str(auc_value) + "\n")
                break

            if epoch_time > 6:
                return 'restart'



        # fit a few more epoch on train+valid until result reaches the best_train_score
        if has_valid and refit:
            if self.greater_is_better:
                best_valid_score = max(self.valid_result)
            else:
                best_valid_score = min(self.valid_result)
            best_epoch = self.valid_result.index(best_valid_score)
            best_train_score = self.train_result[best_epoch]
            Xi_train = Xi_train + Xi_valid
            Xv_train = Xv_train + Xv_valid
            Nv_train = Nv_train + Nv_valid
            y_train = y_train + y_valid
            for epoch in range(100):
                indices_all = np.arange(y_train.shape[0])
                self.shuffle_in_unison_scary(indices_all)
                total_batch = int(indices_all.shape[0] / self.batch_size)

                for i in range(total_batch):
                    indices_selected = self.get_batch(indices_all, self.batch_size, i)  # len(Xi_batch)=1024
                    Xi_batch = Xi_train[indices_selected, :]
                    Xv_batch = Xv_train[indices_selected, :]
                    Nv_batch = Nv_train[indices_selected, :]
                    y_batch = np.array([[y_] for y_ in y_train[indices_selected]])
                    self.fit_on_batch(Xi_batch, Xv_batch, Nv_batch, y_batch)
                # check
                train_result = self.evaluate(Xi_train, Xv_train, Nv_train, y_train)
                valid_result = self.evaluate(Xi_valid, Xv_valid, Nv_valid, y_valid)
                # if abs(train_result - best_train_score) < 0.001 or \
                if abs(valid_result - best_train_score) < 0.001 or \
                        (self.greater_is_better and train_result > best_train_score) or \
                        ((not self.greater_is_better) and train_result < best_train_score):
                    break
        print("AUC", AUC)
        return AUC

    def training_termination(self, valid_result):
        if len(valid_result) > 5:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                        valid_result[-2] < valid_result[-3] and \
                        valid_result[-3] < valid_result[-4] and \
                        valid_result[-4] < valid_result[-5]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                        valid_result[-2] > valid_result[-3] and \
                        valid_result[-3] > valid_result[-4] and \
                        valid_result[-4] > valid_result[-5]:
                    return True
        return False

    def predict(self, Xi, Xv, Nv):
        """
        :param Xi: list of list of feature indices of each sample in the dataset
        :param Xv: list of list of feature values of each sample in the dataset
        :return: predicted probability of each sample
        """
        # dummy y
        dummy_y = np.array([1] * Xi.shape[0])
        indices_all = np.arange(dummy_y.shape[0])
        batch_index = 0
        indices_selected = self.get_batch(indices_all, self.batch_size, batch_index)

        Xi_batch = Xi[indices_selected, :]
        Xv_batch = Xv[indices_selected, :]
        Nv_batch = Nv[indices_selected, :]
        y_batch = np.array([[y_] for y_ in dummy_y[indices_selected]])

        y_pred = None
        while len(Xi_batch) > 0:
            num_batch = y_batch.shape[0]
            feed_dict = {self.feat_index: Xi_batch,
                         self.feat_value: Xv_batch,
                         self.numeric_value: Nv_batch,
                         self.label: y_batch,
                         self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}

            batch_out = self.sess.run(self.out, feed_dict=feed_dict)

            if batch_index == 0:
                y_pred = np.reshape(batch_out, (num_batch,))
            else:
                y_pred = np.concatenate((y_pred, np.reshape(batch_out, (num_batch,))))

            batch_index += 1

            indices_selected = self.get_batch(indices_all, self.batch_size, batch_index)
            Xi_batch = Xi[indices_selected, :]
            Xv_batch = Xv[indices_selected, :]
            Nv_batch = Nv[indices_selected, :]
            y_batch = np.array([[y_] for y_ in dummy_y[indices_selected]])
        return y_pred

    def evaluate(self, Xi, Xv, Nv, y):
        y_pred = self.predict(Xi, Xv, Nv)
        return self.eval_metric(y, y_pred)


class Attention(tf.keras.layers.Layer):
    def __init__(self, output_dim1, output_dim2):
        super(Attention, self).__init__()
        self.key_dense = tf.keras.layers.Dense(output_dim1)
        self.value_dense = tf.keras.layers.Dense(output_dim1)
        self.concat_dense = tf.keras.layers.Dense(output_dim2)
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        self.relu = tf.keras.layers.Softmax(axis=-1)
        self.output_dim2 = output_dim2

    def call(self, inputs):
        # Linear transformations
        x_k, x_v = inputs
        k = self.key_dense(x_k)
        v = self.value_dense(x_v)
        x_concat = tf.concat([k, v], axis=-1)
        attn_scores = self.concat_dense(x_concat)
        attn_weights = self.softmax(attn_scores)
        attn_weights_k, attn_weights_v = tf.split(attn_weights, num_or_size_splits=self.output_dim2, axis=-1)
        attn_weights_k = tf.broadcast_to(attn_weights_k, tf.shape(x_k))
        attn_weights_v = tf.broadcast_to(attn_weights_v, tf.shape(x_v))

        k_output = tf.multiply(attn_weights_k, x_k) + x_k
        v_output = tf.multiply(attn_weights_v, x_v) + x_v
        return k_output, v_output


