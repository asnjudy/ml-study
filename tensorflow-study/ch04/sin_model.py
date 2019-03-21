import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

# 定义前向计算的网络结构
input_data = x  # 输入 x
with tf.variable_scope('hidden1'):
    # 第 1 个隐藏层
    weights = tf.get_variable('weight', shape=[1, 16], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0.0, 1))
    biases = tf.get_variable('bias', shape=[1, 16], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0.0, 1))
    hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)

with tf.variable_scope('hidden2'):
    # 第 2 个隐藏层
    weights = tf.get_variable('weight', shape=[16, 16], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0.0, 1))
    biases = tf.get_variable('bias', shape=[16], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0.0, 1))
    hidden2 = tf.sigmoid(tf.multiply(hidden1, weights) + biases)

with tf.variable_scope('hidden3'):
    # 第 3 个隐藏层
    weights = tf.get_variable('weight', shape=[16, 16], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0.0, 1))
    biases = tf.get_variable('bias', shape=[16], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0.0, 1))
    hidden3 = tf.sigmoid(tf.multiply(hidden2, weights) + biases)

with tf.variable_scope('output_layer'):
    # 输出层
    weights = tf.get_variable('weight', shape=[16, 1], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0.0, 1))
    biases = tf.get_variable('bias', shape=[1], dtype=tf.float32,
                             initializer=tf.random_normal_initializer(0.0, 1))
    net_out = tf.multiply(hidden3, weights) + biases

learning_rate = 0.01

# 定义损失函数
loss = tf.square(net_out - y)

# 采用随机梯度下降的优化函数
opt = tf.train.GradientDescentOptimizer(learning_rate)
train_op = opt.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print('Starting training ...')
    for i in range(1000):
        # 随机生成训练数据点
        train_x = np.random.uniform(0.0, 2 * np.pi)
        train_y = np.sin(train_x)
        sess.run(train_op, feed_dict={x: train_x, y: train_y})

    test_x = 0.234
    test_y = sess.run(net_out, feed_dict={x: test_x, y: np.sin(test_x)})
    print('## test', )
