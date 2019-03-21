import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical

# 读入训练数据
all_df = pd.read_table('./SAE/train.txt', header=None, encoding='utf8', sep=',', skiprows=1)
all_df = all_df.sample(frac=1)

y_data = to_categorical(np.array(all_df['genre']))
all_df = all_df.drop(['genre', 'id'])
x_data = np.array(all_df)

# 参数
learning_rate = 0.01    # 学习率
training_epochs = 20    # 训练迭代周期
batch_size = 256        # 每一批次训练的数据量
display_step = 1        # 是否显示计算过程

# 网络配置参数
n_input = x_data.shape[1]
n_hidden_1 = 1024
n_hidden_2 = 512
n_hidden_3 = 128
n_output = y_data.shape[1]

# 图的输入
X = tf.placeholder(tf.float32, [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),

}
