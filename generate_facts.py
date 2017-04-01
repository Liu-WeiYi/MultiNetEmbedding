# coding: utf-8
import networkx as nx
import sys
import os

file_name = sys.argv[1]
raw_file = open(file_name,'r+')
dir_name = 'tmp'
print('generate training data and store graph into gml...')

if not os.path.exists(dir_name):
    os.mkdir(dir_name)

entity_id_file = open('./%s/node_2_id.txt'%dir_name,'w+')
relation2id_file = open('./%s/edge_2_id.txt'%dir_name,'w+')
train_file = open('./%s/src_dst_edge_fact.txt'%dir_name,'w+')

entity_id = set()
relation_id = set()

graph = nx.Graph(name=file_name)

for line in raw_file.readlines():
    e1,e2 = line.strip().split()
    graph.add_node(e1)
    graph.add_node(e2)
    graph.add_edge(e1,e2)

    relation = str(e1)+'→'+str(e2)
    relation_reverse = str(e2)+'→'+str(e1)
    train_file.write(e1+'\t'+e2+'\t'+relation+'\n')
    train_file.write(e2+'\t'+e1+'\t'+relation_reverse+'\n')
    
    entity_id.add(e1)
    entity_id.add(e2)
    relation_id.add(relation)
    relation_id.add(relation_reverse)

# nx.write_gml(graph,'%s/%s.gml'%(dir_name,file_name))

entity_id = list(entity_id)
relation_id = list(relation_id)

for i in range(len(entity_id)):
    entity_id_file.write(entity_id[i]+'\t'+str(i)+'\n')

for j in range(len(relation_id)):
    relation2id_file.write(relation_id[j]+'\t'+str(j)+'\n')

print('down')

