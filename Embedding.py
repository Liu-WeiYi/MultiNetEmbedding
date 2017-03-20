#!/usr/bin/env python
#coding:utf-8
"""
  Author:  unique_liu --<weiyiliu@us.ibm.com>
  Purpose: 对每一层的网络训练模型
  Created: 03/19/17
"""

import random
import numpy as np
import copy
from prepare import prepare

TestFlag = True

########################################################################
class NetEmbedding:
    """
    @purpose 针对每一个网络，训练适用于该网络的模型
    """

    #----------------------------------------------------------------------
    def __init__(self, dimension, margin, Learning_rate, nbatch, nepoch, error):
        """
        @brief 初始化
        @param dimension 特征空间维数
        @param margin 间隔，用以区分 正/负 样本
        @param Learning_rate 学习率
        @param nbatch 批次
        @param nepoch 重复次数
        @param error Loss门限
        """
        self.__dim = dimension
        self.__margin = margin
        self.__learn_rate = Learning_rate
        self.__nbatch = nbatch
        self.__nepoch = nepoch
        self.__error = error

    #----------------------------------------------------------------------
    def init_all_parameters(self, NodesSet, EdgesList):
        """
        @brief 更新当前模型所需参数
        @param NodesSet 所有层网络的所有节点
        @param EdgesList 当前层网络下的 EdgeList
        
        """
        # init
        self.__factPair = set() # 存储当前网络的所有边
        self.__hitNodes = set() # 存储当前网络的所有点
        self.__ignoreNodes = set() # 存储没有在当前网络中出现的节点

        # update
        for edge in EdgesList:
            self.__factPair.add(edge)
            self.__hitNodes.add(edge[0])
            self.__hitNodes.add(edge[1])

        # identify nodes not in NodesSet
        self.__ignoreNodes = NodesSet - self.__hitNodes
        
        """ 将边和点 甩到 k维 空间上 """
        self.__hitNodesVector = self.__initVectorSpace(self.__dim, self.__hitNodes)
        self.__factPairVector = self.__initVectorSpace(self.__dim, self.__factPair)
        self.__ignoreNodesVector = self.__initVectorSpace(self.__dim,self.__ignoreNodes, assignZero=True)

    #----------------------------------------------------------------------
    def train(self):
        """
        """
        # init
        # loss function value
        loss_value = 0
        # batchsize
        batchsize = int(len(self.__factPair)/self.__nbatch)
        
        # 让我们荡起双桨~
        for epoch in range(self.__nepoch):
            loss_value = 0

            # 划向社会主义的远方~
            for batch in range(self.__nbatch):
                self.tmp_hitNodesVector = copy.deepcopy(self.__hitNodesVector)
                self.tmp_factPairVector = copy.deepcopy(self.__factPairVector)

                # 啊~ 大海啊~ 你好多水~~
                for _ in range(batchsize):
                    """ 1. 随机取出一条边, 并构建 src+edge->dst 表达式 """
                    edge = random.sample(self.__factPair,1)[0]
                    fact = (edge[0], edge, edge[1])

                    """ 2. 从Nodes中随机选择一个节点，用来替换edge中的源 或者 目的节点
                           生成一条伪fact
                    """
                    replaceflag,ill_fact = self.__ill_fact(edge, self.__hitNodes, self.__factPair)

                    """ 3. 定义 Loss Function: { margin + d(fact)-d(ill_fact) }+ """
                    dist_fact = self.__L2_distance(fact)
                    dist_ill_fact = self.__L2_distance(ill_fact)

                    # only consider "≥0" cases!!
                    if dist_fact+self.__margin >= dist_ill_fact:
                        loss_value += dist_fact+self.__margin-dist_ill_fact

                        """ 4. 计算梯度, 并更新对应Value """
                        self.__cal_gradient(fact, ill_fact,replaceflag, self.__hitNodesVector, self.__factPairVector)

                """ 5. 完成一个batch之后，更新坐标"""
                self.__hitNodesVector = self.tmp_hitNodesVector
                self.__factPairVector = self.tmp_factPairVector

            """ 6. 做完所有batch之后，输出当前Loss Function的值 """
            if epoch%200 == 0:
                print("epoch: %d"%epoch, end='\t')
                print("loss value: %f"%loss_value)


    #----------------------------------------------------------------------
    def __initVectorSpace(self, dim, targetList, assignZero=False):
        """
        @brief 将当前targetList中的每一个元素投影到dim维空间中，
               并返回以每一个元素为KEY，dim维空间中的坐标为VALUE的字典
        @param dim 空间维数
        @param targetList 当前列表
        @param assignZero 标志位，为 TRUE 表示直接赋0元素
        @return targetList_Dict
        """
        targetList_dict = {}
        
        for target in targetList:
            target_vector = []

            # randomly assign value to target foreach dim
            for k in range(dim):
                if assignZero is False:
                    target_vector.append(random.uniform(-6/(dim**0.5), 6/(dim**0.5)))
                elif assignZero is True:
                    target_vector.append(0)

            # normalization only for non-ingore case!
            if assignZero is False:
                norm_factor = np.linalg.norm(target_vector)
                target_vector = [tmp/norm_factor for tmp in target_vector]

            # store
            targetList_dict[target] = np.array(target_vector)

        return targetList_dict
    
    #----------------------------------------------------------------------
    def __ill_fact(self, edge, Nodes, Edges):
        """
        @brief 生成一条与edge对应的伪fact
        @param Nodes 当前网络所有 节点
        @param Edges 当前网络所有 边
        @return replaceFlag 替换 源节点('src') OR 目的节点('dst')
                ill_fact 伪fact
        """
        replaceFlag = 0
        replace = random.uniform(-1,1)
        if replace > 0: # 替换目的节点
            replaceFlag = 'dst'
            new_dst = random.sample(Nodes,1)[0]
            replaceFlag = True
            while replaceFlag:
                if (edge[0],new_dst) not in Edges and (new_dst,edge[0]) not in Edges:
                    replaceFlag = False
                else:
                    new_dst = random.sample(Nodes,1)[0]
            ill_fact = (edge[0], edge, new_dst)

        elif replace <= 0: #替换源节点
            replaceFlag = 'src'
            new_src = random.sample(Nodes,1)[0]
            replaceFlag = True
            while replaceFlag:
                if (new_src,edge[1]) not in Edges and (new_src,edge[1]) not in Edges:
                    replaceFlag = False
                else:
                    new_src = random.sample(Nodes,1)[0]
            ill_fact = (new_src, edge, edge[1])

        return replaceFlag, ill_fact
    
    #----------------------------------------------------------------------
    def __L2_distance(self,fact):
        """
        @brief 求fact的二范数距离 (src+edge-dst)^2
        @param fact = (src, edge, dst)
        @return L2_distance
        """
        src,edge,dst = fact

        # find vectors
        src_vector = self.__hitNodesVector[src]
        dst_vector = self.__hitNodesVector[dst]
        edge_vector = self.__factPairVector[edge]

        # calculate Distance
        L2_distance = np.linalg.norm(dst_vector-src_vector-edge_vector,2)

        return L2_distance

    #----------------------------------------------------------------------
    def __cal_gradient(self, fact, ill_fact,replaceflag, NodesVector, EdgeVector):
        """
        @brief 计算梯度, 并更新对应 node 和 edge 的坐标
        @param fact     = (src, edge, dst)
        @param ill_fact = (replace_src, edge, dst) or (src, edge, replace_dst)
        @param replaceflag 标志位 源节点('src')被替换 OR 目的节点('dst')被替换
        """
        # extract src, edge, dst, src'(or:/dst')
        src, edge, dst = fact

        if replaceflag == 'src':
            """ case1. 替换 dst 节点"""
            replace_src = ill_fact[0]
            # 1. update src
            # new_src = src - 2R*(src+edge-dst)
            src_vector = NodesVector[src]-2*self.__learn_rate*(NodesVector[src]+EdgeVector[edge]-NodesVector[dst])

            # 2. update edge
            # new_edge = edge - 2R*(src-replace_src)
            edge_vector = EdgeVector[edge]-2*self.__learn_rate*(NodesVector[src]-NodesVector[replace_src])

            # 3. update dst
            # new_dst = dst + 2R*(src-replace_src)
            dst_vector = NodesVector[dst]+2*self.__learn_rate*(NodesVector[src]-NodesVector[replace_src])

            # 4. update replace_src
            # new_replace_src = replace_src + 2R*(replace_src+edge-dst)
            replace_src_vector = NodesVector[replace_src]+2*self.__learn_rate*(NodesVector[replace_src]+EdgeVector[edge]-NodesVector[dst])

            # 归一化并更新
            self.tmp_hitNodesVector[src] = self.__norm_vector(src_vector)
            self.tmp_factPairVector[edge] = self.__norm_vector(edge_vector)
            self.tmp_hitNodesVector[dst] = self.__norm_vector(dst_vector)
            self.tmp_hitNodesVector[replace_src] = self.__norm_vector(replace_src_vector)

        elif replaceflag == 'dst':
            """ case2. 替换 dst 节点"""
            replace_dst = ill_fact[-1]
            # 1. update src
            # new_src = src + 2R*(dst-replace_dst)
            src_vector = NodesVector[src]+2*self.__learn_rate*(NodesVector[dst]-NodesVector[replace_dst])

            # 2. update edge
            # new_edge = edge + 2R*(dst-replace_dst)
            edge_vector = EdgeVector[edge]+2*self.__learn_rate*(NodesVector[dst]-NodesVector[replace_dst])

            # 3. update dst
            # new_dst = dst + 2R*(src+edge-dst)
            dst_vector = NodesVector[dst]+2*self.__learn_rate*(NodesVector[src]+EdgeVector[edge]-NodesVector[dst])

            # 4. update replace_dst
            # new_replace_dst = replace_dst - 2R*(src+edge-replace_dst)
            replace_dst_vector = NodesVector[replace_dst]-2*self.__learn_rate*(NodesVector[src]+EdgeVector[edge]-NodesVector[replace_dst])

            # 归一化并更新
            self.tmp_hitNodesVector[src] = self.__norm_vector(src_vector)
            self.tmp_factPairVector[edge] = self.__norm_vector(edge_vector)
            self.tmp_hitNodesVector[dst] = self.__norm_vector(dst_vector)
            self.tmp_hitNodesVector[replace_dst] = self.__norm_vector(replace_dst_vector)

    #----------------------------------------------------------------------
    def __norm_vector(self,vector):
        """
        @brief 归一化向量
        @param vector
        @return vector(已归一化~)
        """
        norm_factor = np.linalg.norm(vector)
        vector = [_/norm_factor for _ in norm_factor]
        return vector

if TestFlag == True:
    if __name__ == "__main__":
        folder_dir = './Edge_Lists/AUCS/'
        """
        =======================
        prepare 数据
        =======================
        """
        DirectFlag = False
        NodesSet, EdgesList_layers = prepare(folder_dir,DirectFlag)
        
        """
        =======================
        设置超参数
        =======================
        """
        dimension = 10
        margin = 1
        Learning_rate = 0.001
        nbatch = 1
        nepoch = 1000
        error = 0.0000001
        """
        =======================
        Embedding 每一个网络
        =======================
        """
        for net in EdgesList_layers.keys():
            print('current Nets: %s'%net)
            NetSpace = NetEmbedding(dimension, margin, Learning_rate, nbatch, nepoch, error)
            NetSpace.init_all_parameters(NodesSet, EdgesList_layers[net])
            NetSpace.train()