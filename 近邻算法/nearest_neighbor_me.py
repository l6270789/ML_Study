'''
Created on 2017年4月18日

@author: mx

nearest_neighbor
'''

import tensorflow as tf
import numpy as np

#导入mnist数据
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("data/mnist",one_hot=True)

#选取训练集、测试集数目
X_train,Y_train=mnist.train.next_batch(50000)
X_test,Y_test=mnist.test.next_batch(500)

#定义变量大小（）
xtr=tf.placeholder("float", [None,784])
xte=tf.placeholder("float", [784])


#计算测试数据与训练数据L1范数大小（1表示从横轴进行降维）

distance=tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte, ))), 1)

#求得distance最小的下标（0表示从竖轴计算）
predict=tf.arg_min(distance, 0)

#准确率初始0

Accuracy=0


#数据初始化
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#开始预测
for i in range(len(X_test)):
   
    #近邻算法:测试集与训练集对比，返回误差最小的下标
    nn_index=sess.run(predict,feed_dict={xtr:X_train,xte:X_test[i,:]})
    
    #np.argmax  返回标签Y中最大数下标（既数值为1的下标），也就是该标签所对应的数字
    print("Test :",i,"Prection :",np.argmax(Y_train[nn_index]),"True class :",np.argmax(Y_test[i]))
    
    #统计准确率
    if np.argmax(Y_train[nn_index])==np.argmax(Y_test[i]):
        
        Accuracy+=1/len(X_test)
        
print("Accuracy :",Accuracy)
    
    
    
    
    
    
    

