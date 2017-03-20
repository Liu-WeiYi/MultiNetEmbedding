#!/usr/bin/env python
#coding:utf-8
"""
  Author:  unique_liu --<weiyiliu@us.ibm.com>
  Purpose: 
  Created: 03/19/17
"""

import sys, glob, os

TestFlag = True


#----------------------------------------------------------------------
def prepare(folder_dir,DirectFlag=False):
    """
    @brief 准备工作~
    @param folder_dir 存放Edge_list的路径
    @param DirectFlag 当前存储的图是否为有向图, True - 有向 / False - 无向
    @return 
    """
    
    """ 1. Init """
    edge_list_files_dir = glob.glob(folder_dir+"*")
    NodesSet = set()
    EdgesList_layers = {}
    
    for dir in edge_list_files_dir:
        # 对每一层Layer，进行初始化
        edge_list = open(dir,'r')
        layer_name = dir
        EdgesList = []
        for line in edge_list.readlines():
            try:
                src,dst = line.strip().split()
            except:
                print('格式有问题...')

            # update NodesID Set
            NodesSet.add(src)
            NodesSet.add(dst)
            # update EdgesID List
            edge = (src,dst)
            if edge not in EdgesList:
                EdgesList.append(edge)
            if DirectFlag is False:
                reversed_edge = (dst,src)
                EdgesList.append(reversed_edge)
        
        # update dicts
        EdgesList_layers[dir] = EdgesList
        
        edge_list.close()
    
    """2. 输出每层网络信息"""
    print('总共的Nodes个数: %d'%len(NodesSet))
    for net in EdgesList_layers.keys():
        print('current layer: %s'%net + '\tEdges Number: %d'%len(EdgesList_layers[net]))
    
    print('Prepare Down...')
    
    return NodesSet, EdgesList_layers



if TestFlag == True:
    if __name__ == "__main__":
        folder_dir = './Edge_Lists/AUCS/'
        prepare(folder_dir)