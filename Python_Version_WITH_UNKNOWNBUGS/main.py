#!/usr/bin/env python
#coding:utf-8
"""
  Author:  uniqueliu --<weiyiliu@us.ibm.com>
  Purpose: 程序调用入口
  Created: 03/20/17
"""
from Embedding import *
import sys
import pickle
if __name__ == "__main__":
    
    #folder_dir = './Edge_Lists/AUCS/'
    folder_dir = './Edge_Lists/%s/'%sys.argv[1]
    save_path = './results/%s/'%sys.argv[1]
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
    dimension = int(len(NodesSet)/100)
    margin = 1
    Learning_rate = 0.0001
    nbatch = 1
    nepoch = 1000
    error = 0.0000001
    print('Dim: %d'%dimension)
    print('LR: %f'%Learning_rate)
    """
    =======================
    Embedding 每一个网络
    =======================
    """
    net_name_list = []
    for net in EdgesList_layers.keys():
        print('current Nets: %s'%net)
        net_name = net.split('/')[-1]
        net_name_list.append(net_name)
        NetSpace = NetEmbedding(dimension, margin, Learning_rate, nbatch, nepoch, error)
        NetSpace.init_all_parameters(NodesSet, EdgesList_layers[net])
        NetSpace.train()

        # 保存。。。
        NodesVector = NetSpace.hitNodesVector
        ignoreVector = NetSpace.ignoreNodesVector
        EdgesVector = NetSpace.EdgeVector

        info = [NodesVector, ignoreVector, EdgesVector]
        pickle.dump(info,open(save_path+"%s.pickle"%net_name,'wb'))
    pickle.dump(net_name_list,open(save_path+"net_index.pickle","wb"))
