'''
Created on 2017年4月24日

@author: mx
'''
import tensorflow as tf

# 导入数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/mnist", one_hot=True)

# 定义参数:学习率;迭代次数;批处理个数;显示步长
learning_rate = 0.01
iterations = 15
batch_number = 100
display_step = 1

# 定义输入输出矩阵大小
X = tf.placeholder("float", [None, 784])  # 28*28
Y = tf.placeholder("float", [None, 10])  # 10 labels  : 0-9

# 定义隐藏层大小
hidden_one = 256
hidden_two = 256

# 定义多重感知机函数，需要输入参数X，权重weights，偏置biases
# 这里的weights和biases是python中一个字典， dict = {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}
# dict['Alice']=2341
def multilayer_perceptron(x, weights, biases):
    hidden_layer_one = tf.matmul(x, weights["W_layer_one"]) + biases["b_layer_one"]
    
    # tf.nn.relu()属于激活函数，几种常见的激活函数如下：
    # Sigmoid（S 型激活函数）：输入一个实值，输出一个 0 至 1 间的值 σ(x) = 1 / (1 + exp(-x))
    # tanh（双曲正切函数）：输入一个实值，输出一个 [-1,1] 间的值 tanh(x) = 2σ(2x)-1
    # ReLU：ReLU 代表修正线性单元。输出一个实值，并设定 0 的阈值（函数会将负值变为零）f(x) = max(0, x)
    hidden_layer_one = tf.nn.relu(hidden_layer_one)
    hidden_layer_two = tf.matmul(hidden_layer_one, weights["W_layer_two"]) + biases["b_layer_two"]
    hidden_layer_two = tf.nn.relu(hidden_layer_two)
    
    # 返回隐藏层2的输出，也就是输出层的输入，这里不需要求最后的输出，只需返回没有经过激活函数的输入，后面会讲原因
    out_layer = tf.matmul(hidden_layer_two, weights["W_out_layer"]) + biases["b_out_layer"]
    return out_layer


# tf.random_normal():随机生成均值为0，标准差为1的数值
weights = {"W_layer_one":tf.Variable(tf.random_normal([784, hidden_one])),
         "W_layer_two":tf.Variable(tf.random_normal([hidden_one, hidden_two])),
         "W_out_layer":tf.Variable(tf.random_normal([hidden_two, 10]))   
    }
biases = {"b_layer_one":tf.Variable(tf.random_normal([hidden_one])),
        "b_layer_two":tf.Variable(tf.random_normal([hidden_two])),
        "b_out_layer":tf.Variable(tf.random_normal([10]))  
    } 

# 调用多重感知机函数
pred = multilayer_perceptron(X, weights, biases)

# tf.nn.softmax_cross_entropy_with_logits(): 
# http://www.jianshu.com/p/fb119d0ff6a6(对这个函数的说明)
# 这个函数可以利用softmax函数对输入进行映射，并求出与真实值之间的交叉熵
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred))

# 另外一种优化函数，在这个算法中比梯度下降优化更快
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 初始化参数
init = tf.global_variables_initializer()   
sess = tf.Session()
sess.run(init)

for iteration in range(iterations):
    
    # 初始训练误差，计算每轮批量迭代次数
    train_cost=0
    batch_times=int(mnist.train._num_examples/batch_number)
    
    for i in range(batch_times):
       
        # 每次取100张图
        batch_X,batch_Y=mnist.train.next_batch(batch_number)
        
        # 运行优化函数
        # 这里返回一个[optimizer,cost]的list, 其中 _代表optimizer,batch_cost代表cost的值
        _,batch_cost=sess.run([optimizer,cost],feed_dict={X:batch_X,Y:batch_Y})
        
        # 返回训练集误差：每次计算100张图的batch_cost，计算了i次，所以最后除以batch_numbers
        train_cost+=batch_cost/batch_times
        
    if iteration % display_step==0:
        
        # %04d: % 转义说明符 ; 0 指以0填充前面的位数 ；4 四位数； d 十进制整数
        # "{:.9f}".format(train_cost)  以保留小数点后9位显示train_cost
        print("Iteration :","%04d"%(iteration+1),"Train_cost :","{:.9f}".format(train_cost))
 
# tf.arg_max(pred,1):得到向量中最大数的下标，1代表水平方向
# tf.equal():返回布尔值，相等返回1，否则0
# 最后返回大小[none,1]的向量，1所在位置为布尔类型数据       
prediction=tf.equal(tf.arg_max(pred, 1), tf.arg_max(Y,1))    

# tf.cast():将布尔型向量转换成浮点型向量
# tf.reduce_mean():求所有数的均值
# 返回正确率：也就是所有为1的数目占所有数目的比例
accuracy=tf.reduce_mean(tf.cast(prediction,"float"))

# 打印正确率
print("Train_cost :",sess.run(accuracy,feed_dict={X:mnist.train.images,Y:mnist.train.labels}))
print("Test_cost :",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))


# 打印结果
# Iteration : 0001 Train_cost : 53.053542605
# Iteration : 0002 Train_cost : 8.394692302
# Iteration : 0003 Train_cost : 4.178018635
# Iteration : 0004 Train_cost : 2.850156586
# Iteration : 0005 Train_cost : 2.115638643
# Iteration : 0006 Train_cost : 2.068853149
# Iteration : 0007 Train_cost : 1.676146167
# Iteration : 0008 Train_cost : 1.487952944
# Iteration : 0009 Train_cost : 1.493097927
# Iteration : 0010 Train_cost : 1.245084499
# Iteration : 0011 Train_cost : 1.224542024
# Iteration : 0012 Train_cost : 0.930452932
# Iteration : 0013 Train_cost : 0.838671544
# Iteration : 0014 Train_cost : 0.777738831
# Iteration : 0015 Train_cost : 0.635424245
# Train_cost : 0.984127
# Test_cost : 0.9646




