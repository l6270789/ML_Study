'''
Created on 2017年4月23日

@author: mx
'''

import tensorflow as tf

# 导入mnist数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/temp/data/",one_hot=True)

# 定义一些常量
learning_rate=0.01
epoch_times=25
display_step=1

# 定义批量梯度下降次数，每100张图计算一次梯度
batch_size=100

# 定义X,Y大小  None代表第几张图,784=28*28,10 classes
# Y=[[1,0,0...,0],
#    [0,0,1...,0],
#    [0,1,0...,0],
#    [0,0,0...,1],
#    ...........]
X=tf.placeholder("float", [None,784])
Y=tf.placeholder("float", [None,10])


# 定义W,b大小
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))


# matmul()矩阵相乘  ;multiply()实数相乘
# sigmoid将一个real value映射到（0,1）的区间（当然也可以是（-1,1）），这样可以用来做二分类。 
# softmax把一个k维的real value向量（a1,a2,a3,a4….）映射成一个（b1,b2,b3,b4….）
# 其中b是一个0-1的常数，然后可以根据bi的大小来进行多分类的任务，如取权重最大的一维。 

# 返回一个10维矩阵 
# 注意X,W前后顺序 [None,784]*[784,10]=[None,10]
pred=tf.nn.softmax(tf.matmul(X, W)+b)


# -Y*tf.log(pred)：交叉熵代价函数 
# 代价函数接近于0.(比如Y=0，pred～0；Y=1，pred~1时，代价函数都接近0)
# 这里的*指对应下标的数相乘，不属于向量相乘，因此返回矩阵大小仍为[None,10]


# tf.reduce_sum(-Y*tf.log(pred),1): 
# 返回每个实例的交叉熵（向量），1代表从水平方向求和
# tf.reduce_mean():返回所有交叉熵的平均值（实数）
cost= tf.reduce_mean(tf.reduce_sum(-Y*tf.log(pred),1))

# 调用梯度下降函数
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化参数方法
init=tf.global_variables_initializer()

# 运行参数初始化方法
sess=tf.Session()
sess.run(init)


for epoch in range(epoch_times):
    
    # 初始训练集误差为0
    train_cost=0
    
    # 批量梯度下降，返回总批量次数（55000/100=550）
    batch_numbers=int(mnist.train.num_examples/batch_size)
    
    for i in range(batch_numbers):
        
        # 每次取100张图片
        batch_Xs,batch_Ys=mnist.train.next_batch(batch_size)
        
        # 运行优化函数
        # 这里返回一个[optimizer,cost]的list, 其中 _代表optimizer,batch_cost代表cost的值
        _,batch_cost=sess.run([optimizer,cost],feed_dict={X:batch_Xs,Y:batch_Ys})
        
        # 等价于上面
        # sess.run(optimizer,feed_dict={X:batch_Xs,Y:batch_Ys})
        # batch_cost=sess.run(cost,feed_dict={X:batch_Xs,Y:batch_Ys})
        
        # 返回训练集误差：每次计算100张图的batch_cost，计算了i次，所以最后除以batch_numbers
        train_cost+=batch_cost/batch_numbers
    
    
    # 打印每次迭代的误差
    if (epoch+1)%display_step==0:
        
        # %04d: % 转义说明符 ; 0 指以0填充前面的位数 ；4 四位数； d 十进制整数
        # "{:.9f}".format(train_cost)  以保留小数点后9位显示train_cost
        print("Epoch :","%04d"%(epoch+1),"Train_cost","{:9f}".format(train_cost))
    
print("Optimization finished!")


# tf.arg_max(pred,1):得到向量中最大数的下标，1代表水平方向
# tf.equal():返回布尔值，相等返回1，否则0
# 最后返回大小[none,1]的向量，1所在位置为布尔类型数据
correct_prediction=tf.equal(tf.arg_max(pred,1),tf.arg_max(Y,1))

# tf.cast():将布尔型向量转换成浮点型向量
# tf.reduce_mean():求所有数的均值
# 返回正确率：也就是所有为1的数目占所有数目的比例
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

# 打印正确率
print("Train_accuracy :",sess.run(accuracy,feed_dict={X:mnist.train.images,Y:mnist.train.labels}))
print("Test_accuracy :",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))




















