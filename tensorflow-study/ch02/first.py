import tensorflow as tf


def tensor_trainable():
    weight1 = tf.Variable(0.001, name='weight_1')
    weight2 = tf.Variable(weight1.initialized_value() * 2, name='weight_2', trainable=False)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        print('weight1 is:', sess.run(weight1))
        print('weight2 is:', sess.run(weight2))


def tensor_reshape():
    v = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9], name='v_1')
    reshaped_v = tf.reshape(v, [3, 3])

    print(v.shape)
    print(reshaped_v.shape)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(v))
        print(sess.run(reshaped_v))


def tensor_shape():
    value0 = tf.Variable(754)
    value1 = tf.Variable([1.1, 2, 4, 5, 6.66])
    value2 = tf.Variable([[1, 2], [3, 4], [4, 5]])

    print(value0.shape)
    print(value1.shape)
    print(value2.shape)


def variable_scope_reuse_variables():
    """
    通过 tf.variable_scope('inference') 声明变量空间
    通过 tf.get_variable('weight') 创建变量并重用变量
    通过 tf.get_variable_scope().reuse_variables() 设置变量可重用
    :return:
    """
    x = tf.placeholder(tf.float32, name='train_x')
    with tf.variable_scope('inference'):
        # 还未开启变量重用
        # 在 inference 空间下，创建变量 weight, bias
        weight = tf.get_variable('weight', [1], initializer=tf.constant_initializer(2))
        bias = tf.get_variable('bias', [1], initializer=tf.constant_initializer(0.015))
        y = x * weight + bias
        print('### weight:', weight)
        print('### bias:', bias)
        print('### y:', y)

        # 启用当前作用域下的变量重用功能
        tf.get_variable_scope().reuse_variables()
        # 下面的 weight, bias 必须在前面已定义，否则报错
        weight2 = tf.get_variable('weight', [1])
        bias2 = tf.get_variable('bias', [1])
        y2 = x * weight + bias

        print('### weight2:', weight2)
        print('### bias2:', bias2)
        print('### y2:', y2)

        init = tf.global_variables_initializer()
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(init)
            print(sess.run(weight))
            print(sess.run(weight2))
            print(sess.run(bias))
            print(sess.run(bias2))
            print(sess.run(y, feed_dict={x: 2}))
            print(sess.run(y2, feed_dict={x: 2}))


if __name__ == '__main__':
    v1 = tf.placeholder(tf.float32)
    v2 = tf.placeholder(tf.float32)
    v_mul = tf.multiply(v1, v2)
    b = tf.constant(2.5, name='bias')
    v_add = tf.add(v_mul, b)

    with tf.Session() as sess:
        print('result:', sess.run(v_add, feed_dict={v1: 13, v2: 3}))

# variable_scope_reuse_variables()
