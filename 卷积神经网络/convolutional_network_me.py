'''
Created on 2017年4月27日

@author: mx
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/mnist", one_hot=True)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])


# 根据张量大小shape随机生成权重W的函数
def weight_variable(shape):
    """
    tf.truncated_normal与tf.random_normal区别
    http://blog.csdn.net/u013713117/article/details/65446361
    """
    # 在tf.truncated_normal中如果x的取值在区间（μ-2σ，μ+2σ）之外则重新进行选择。
    # 保证了生成的值都在均值附近。
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 根据张量大小shape固定生成偏置b=0.1的函数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 卷积函数
def conv2d(x, W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
          前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

# 池化函数
def max_pool_2x2(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
          几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

"""
卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。
即总的卷积输出应该为None*28*28*32,也就是单个通道输出为28*28，共有32个通道,共有None个批次
在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是None*14*14*32
"""
# 第一场卷积层
W_conv_one = weight_variable([5, 5, 1, 32])
b_conv_one = bias_variable([32])

"""
 -1 can also be used to infer the shape
  # -1 is inferred to be 9:
  reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 2:
  reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
  # -1 is inferred to be 3:
  reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]],
                               [[4, 4, 4],
                                [5, 5, 5],
                                [6, 6, 6]]]
"""
# 将输入按照 conv2d中input的格式来reshape
X_images = tf.reshape(X, [-1, 28, 28, 1])

# tf.nn.relu()属于激活函数，几种常见的激活函数如下：
# Sigmoid（S 型激活函数）：输入一个实值，输出一个 0 至 1 间的值 σ(x) = 1 / (1 + exp(-x))
# tanh（双曲正切函数）：输入一个实值，输出一个 [-1,1] 间的值 tanh(x) = 2σ(2x)-1
# ReLU：ReLU 代表修正线性单元。输出一个实值，并设定 0 的阈值（函数会将负值变为零）f(x) = max(0, x)
h_conv_one = tf.nn.relu(conv2d(X_images, W_conv_one) + b_conv_one)
h_pool_one = max_pool_2x2(h_conv_one)


"""
# 卷积核5*5，输入通道为32，输出通道为64。
# 卷积前图像的尺寸为 None*14*14*32， 卷积后为None*14*14*64
# 池化后，输出的图像尺寸为None*7*7*64
"""
# 第二层卷积层
W_conv_two = weight_variable([5, 5, 32, 64])
b_conv_two = bias_variable([64])

h_conv_two = tf.nn.relu(conv2d(h_pool_one, W_conv_two) + b_conv_two)
h_pool_two = max_pool_2x2(h_conv_two)

# 全连接层
# 输入维数7*7*64, 输出维数为1024
W_fc_one = weight_variable([7 * 7 * 64, 1024])
b_fc_one = bias_variable([1024])

h_pool_two_reshape = tf.reshape(h_pool_two, [-1, 7 * 7 * 64])
h_fc_one = tf.nn.relu(tf.matmul(h_pool_two_reshape, W_fc_one) + b_fc_one)

# 这里使用了dropout,即随机安排一些cell输出值为0，可以防止过拟合
# keep_prob:probability that a neuron's output is kept during dropout
# 训练的时候取0.5防止过拟合，验证的时候取1.0保证正确率
keep_prob = tf.placeholder("float")
h_fc_one_drop = tf.nn.dropout(h_fc_one, keep_prob)

# 输出层
# 输入1024维，输出10维，也就是具体的0~9分类
W_out = weight_variable([1024, 10])
b_out = bias_variable([10])

# 最终输出
pred = tf.matmul(h_fc_one_drop, W_out) + b_out

# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred))

# 优化函数
optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)

# 预测函数
prediction = tf.equal(tf.arg_max(pred, 1), tf.arg_max(Y, 1))

# 计算准确率函数
accuracy = tf.reduce_mean(tf.cast(prediction, "float"))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={X:batch[0], Y:batch[1], keep_prob:1.0})
        print("Iterations :", i, "Train_accuracy :", "{:.9f}".format(train_accuracy))
    sess.run(optimizer,feed_dict={X:batch[0], Y:batch[1], keep_prob:0.5})
    
print("Train_accuracy :", sess.run(accuracy, feed_dict={X:mnist.train.images[:256],
                                                      Y:mnist.train.labels[:256], keep_prob:1.0}))
print("Test_accuracy :", sess.run(accuracy, feed_dict={X:mnist.test.images[:256],
                                                      Y:mnist.test.labels[:256], keep_prob:1.0}))

