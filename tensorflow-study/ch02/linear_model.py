import numpy as np
import tensorflow as tf


def get_data(number):
    list_x = []
    list_label = []

    for i in range(number):
        x = np.random.randn(1)
        label = 2 * x + np.random.randn(1) * 0.01 + 10
        list_x.append(x)
        list_label.append(label)

    return list_x, list_label


class SimpleLinearModel:

    def __init__(self):
        self.train_x = tf.placeholder(tf.float32, name='train_x')
        self.train_label = tf.placeholder(tf.float32, name='train_label')

        self.test_x = tf.placeholder(tf.float32, name='test_x')
        self.test_label = tf.placeholder(tf.float32, name='test_label')

        # 读取数据
        self.train_data_x, self.train_data_label = get_data(1000)
        self.test_data_x, self.test_data_label = get_data(1)

        self.model_path = './model/linear_model.ckpt'

    def inference(self, x):
        weight = tf.get_variable('weight', [1])
        bias = tf.get_variable('bias', [1])
        y = x * weight + bias
        return y

    def train(self, save_model=True):
        with tf.variable_scope('inference'):
            train_y = self.inference(self.train_x)
            tf.get_variable_scope().reuse_variables()
            test_y = self.inference(self.test_x)

        train_loss = tf.square(train_y - self.train_label)
        test_loss = tf.square(test_y - self.test_label)

        opt = tf.train.GradientDescentOptimizer(0.002)
        train_op = opt.minimize(train_loss)
        # 初始化所有变量
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 开始训练
            for i in range(1000):
                # 运行一次训练操作
                sess.run(train_op, feed_dict={
                    self.train_x: self.train_data_x[i],
                    self.train_label: self.train_data_label[i]
                })

                if i % 10 == 0:
                    # 每运行 10 次训练，测试一下数据
                    test_loss_value = sess.run(test_loss, feed_dict={
                        self.test_x: self.test_data_x[0],
                        self.test_label: self.test_data_label[0]
                    })
                    print('step %d eval loss is %.3f' % (i, test_loss_value))

            if save_model:
                # 模型保存 - 把训练好的参数保存起来
                saver = tf.train.Saver()

                save_path = saver.save(sess, self.model_path)
                print('Model saved in file: %s' % save_path)

            # test, predict
            print('## 100:', sess.run(train_y, feed_dict={self.train_x: 139}))

    def predict(self, x):
        with tf.variable_scope('inference'):
            train_y = self.inference(self.train_x)

        # test
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('Loading model parameters ...')
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)

            print('## 100:', sess.run(train_y, feed_dict={self.train_x: x}))


if __name__ == '__main__':
    model = SimpleLinearModel()
    # model.train(save_model=True)
    model.predict(139)
