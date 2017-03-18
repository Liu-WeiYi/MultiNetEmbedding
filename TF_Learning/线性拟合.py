#!/usr/bin/env python
#coding:utf-8
"""
  Purpose: 学习TF的线性拟合方式
  Created: 03/18/17
"""
import tensorflow as tf
import numpy as np

# ============================================================ #
# 构造一个线性函数 Y = 0.1·X1 + 0.2·X2 + 0.3
# ============================================================ #

x_data = np.float32(np.random.rand(2,100))
"""
1. np.random.rand(row, column)
    --- np的筛选函数，表示筛选一个2行，100列的矩阵
         即: 随机生成两个向量，每一个向量一共100维
    --- 其中每一个元素的范围都在 [0,1）上
2. np.float32(矩阵)
    --- 只是将矩阵中的每一个值都变成一个float32类型
"""

y_data = np.dot([0.100,0.200], x_data) + 0.300
"""
1. np.dot(向量的每一个元素上的值，其大小应该与当前向量的行数有关， 向量)
    --- 举例:
    X =
        np.array([[ 0.1,  0.5],
                  [ 0.2,  0.6]])
    W = np.array([10,20])
    
    若
    Y = np.dot(W,X)
    则表明Y = [10*0.1+20*0.2, 10*0.5+20*0.6] = [5, 17]
    ∴ Y = 
        array([  5.,  17.])
"""

# 综上所述，上面两行描述的是一个线性函数: Y = 0.1·X1 + 0.2·X2 + 0.3

# ============================================================ #
# 构造线性模型 ---> 特点: 所有的变量、表达式都是用tf来存储
# ============================================================ #

b = tf.Variable(tf.zeros([1]))
"""
1. tf.zeros(shape) 
    --- 生成一个全0矩阵，其矩阵的大小及为shape值
    --- 比如在这里，就生成一个全0矩阵
    --- 值得注意的是，我们只说明了其shape为1，
    --- 表明实际上我们就生成一个只有一个元素的矩阵 array([ 0.])
2. tf.Variable()
    --- 这个为TF自己的类 Variable
    --- 用来存放初始化的变量
    --- 在这里，将一个元素的矩阵放入tf.Variable中，表示了b元素的初始值
"""
W = tf.Variable(tf.random_uniform([1,2],-0.1,1.0))
"""
1. random_uniform([shape],minValue,maxValue) 
    --- tf的均匀分布函数，从给定区间 [minValue,maxValue）产生一个随机数
    --- shape: 矩阵大小 表示生成一个 1行 2列的矩阵
    --- 在这里，表示生成一个1行2列的矩阵，其中每一个元素在[-0.1，0.1]之间
2. tf.Variable()
    --- 这里，“告诉”TF如何初始化W矩阵
    --- 具体的，W矩阵应该被初始化为一个二维向量，其中每一个元素都在[0.1,1.0]之间
"""
y = tf.matmul(W,x_data)+b
"""
1. tf.matmul(A,B)
    ---  矩阵A和B相乘
"""
# 定义 Loss Function
loss = tf.reduce_mean(tf.square(y-y_data))
"""
来源: http://blog.csdn.net/qq_32166627/article/details/52734387
(相信这个说的很清楚了！)
tensorflow中有一类在tensor的某一维度上求值的函数。如：
    求最大值tf.reduce_max(input_tensor, reduction_indices=None, keep_dims=False, name=None)
    求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
        参数（1）input_tensor:待求值的tensor。
        参数（2）reduction_indices:在哪一维上求解。
        参数（3）（4）可忽略
    举例说明：
        'x' is [[1., 2.]
                [3., 4.]]
        x是一个2维数组，分别调用reduce_*函数如下：
    首先求平均值，
        tf.reduce_mean(x) ==> 2.5 
        #如果不指定第二个参数，那么就在所有的元素中取平均值
        tf.reduce_mean(x, 0) ==> [2.,  3.] 
        #指定第二个参数为0，则第一维的元素取平均值，即每一列求平均值
        tf.reduce_mean(x, 1) ==> 
        [1.5,  3.5] #指定第二个参数为1，则第二维的元素取平均值，即每一行求平均值
    同理，还可用tf.reduce_max()求最大值。
"""
#  指定优化方法
optimizer = tf.train.GradientDescentOptimizer(0.5)
# 指定训练目标
train = optimizer.minimize(loss)
"""
这里的优化目标是最小化Loss Function
"""

# ============================================================ #
# 启动TF
# ============================================================ #

# 1. 初始化所有变量
init = tf.global_variables_initializer()

# 2. 启动
sess = tf.Session()
sess.run(init)

step = 0
# 拟合
while sess.run(loss) > 0.00000000001 or step > 1000:
    sess.run(train)
    print('step: %d'%step,end = '\t')
    print('W: %s'%str(sess.run(W)),end = '\t')
    print('b: %s'%str(sess.run(b)),end = '\t')
    print('Loss: %f'%sess.run(loss))
    step += 1
