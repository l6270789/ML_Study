'''
Created on 2017年4月21日

@author: mx
'''

import tensorflow as tf
import matplotlib.pyplot as plt


#导入numpy中生成随机数的包
import numpy.random as npr
import numpy as np


#设置学习率、迭代次数、每多少迭代次数计算一次误差
#学习率由于训练数据原因，此处要设置较小，否则梯度下降不起作用
learning_rate=0.01
epoch_times=1000
epoch_display=50

#设置训练集
X_train=np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y_train=np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])



#初始化参数
x=tf.placeholder("float")
y=tf.placeholder("float")

W=tf.Variable(npr.randn(),name="weight")
b=tf.Variable(npr.randn(),name="bias")

#计算  W*x+b
pred=tf.add(tf.multiply(W, x),b)

#计算均方误差   shape[0]读取数组第一维的长度（0表示第一维）
cost=tf.reduce_sum(tf.pow(pred-y, 2))/(2*X_train.shape[0])

#调用优化函数
optimizer=tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#参数初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#计算训练集误差


for epoch in range(epoch_times):
    
#     zip():
#     a = [1,2,3]
#     b = [4,5,6]
#     使用zip()函数来可以把列表合并，并创建一个元组对的列表。
#     zip(a,b)
#     [(1, 4), (2, 5), (3, 6)]
    
#     for (X,Y) in zip(X_train,Y_train):
   
    #运行优化函数
    sess.run(optimizer,feed_dict={x:X_train,y:Y_train})
    
    #每多少次计算一次训练集误差
    if (epoch+1) % epoch_display==0:
        
        train_cost=sess.run(cost,feed_dict={x:X_train,y:Y_train})
        
        #%04d: % 转义说明符 ; 0 指以0填充前面的位数 ；4 四位数； d 十进制整数
        #"{:.9f}".format(train_cost)  以保留小数点后9位显示train_cost
        print("Epoch :","%04d"%(epoch+1),"cost :","{:.9f}".format(train_cost),
              "W =",sess.run(W),"b =",sess.run(b))
    
print("Train_cost :",train_cost,"W =",sess.run(W),"b =",sess.run(b))

#ro:红色点状  bo：蓝色点状
plt.plot(X_train,Y_train,"ro",label="Original Data")
plt.plot(X_train,sess.run(W) * X_train + sess.run(b),label='Fitted Line')

#添加图例
plt.legend()
plt.show()

#设置测试数据    
X_test=np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1]) 
Y_test=np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])   

#计算测试数据误差,由于数据个数不同 不能直接调用sess.run(cost)
Test_cost=sess.run(tf.reduce_sum(tf.pow(pred-y,2))/(2*X_test.shape[0]),
                   feed_dict={x:X_test,y:Y_test})

print("Test_cost :",Test_cost)



