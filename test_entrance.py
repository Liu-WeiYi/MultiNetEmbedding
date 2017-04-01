# Sailung Yeung
# <yeungsl@bu.edu>
# reference:
# http://www.cnblogs.com/edwardbi/p/5509699.html

import numpy as np
import Node2Vec
import Word2Vec
import linking_test
import collections, math, os, random, http.client, sys, time

# Read in the same data as used in tensoflow template
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
# for comparasion


# Read the data into a list of strings.
def read_data(filename):
  """Extract the file and convert it into a neighbor list"""
  BFSlist = {}
  Edgelist = []
  for line in open(filename):
    s, d = line.split(" ")
    src = int(s)
    dst = int(d)
    Edgelist.append((src, dst))
    if src not in list(BFSlist.keys()):
      BFSlist[src] = {dst: 1}
      if dst not in list(BFSlist.keys()):
        BFSlist[dst] = {src: 1}
      else:
        BFSlist[dst].update({src: 1})
    else:
      BFSlist[src].update({dst: 1})
      if dst not in list(BFSlist.keys()):
        BFSlist[dst] = {src: 1}
      else:
        BFSlist[dst].update({src: 1})
  return BFSlist, Edgelist

################# test main code ###########################################
# python3 test_entrance <filename> 
# [<p> <tmp/node_2_id> <tmp/filename.node>] --- 不需要

start = time.time()
origin_filename = sys.argv[1]
# origin_filename = "2004-04"
results_file = open('%s_results.txt'%origin_filename,'w+')
results_file.write('P'+'\t'+'NODE2VEC_P'+'\t'+'NODE2VEC_AUC'+'\t'+'Embedding_P'+'\t'+'Embedding_AUC'+'\n')

BFSlist, Edgelist = read_data(origin_filename)

print(('edge list size', len(Edgelist)))
for pp in range(1,50,1):
  # T = linking_test.Test(BFSlist, Edgelist, float(sys.argv[2]))
  # p = 0.1
  pp = pp/100
  print('current p is ', pp)
  results_file.write(str(pp)+'\t')

  T = linking_test.Test(BFSlist, Edgelist, pp)
  Removelist, New_BFSlist, New_Edgelist = T.sample()
  print(('New edge list size', len(New_Edgelist)))

  filename = origin_filename+"_sample"
  with open(filename,'w+') as f:
    for edge in Edgelist:
      f.write(str(edge[0])+'\t'+str(edge[1])+'\n')
    
  # --------------------------------------------------------------------------------- #
  # Node2Vec
  # generating all the walks that needed in learning
  p = 0.5
  q = 0.5
  num_walks = 10
  walk_length = 80

  G = Node2Vec.Graph(New_BFSlist, New_Edgelist, p, q)
  G.preprocess_transition_probs()
  walks = G.simulate_walks(num_walks, walk_length)
  print('walk list size', len(walks))
  words = []
  for walk in walks:
    words.extend([str(step) for step in walk])


  L = Word2Vec.Learn(words)
  matrix, mapping = L.train()
  percentage, AUC = T.run_test(Removelist, matrix, mapping, BFSlist)
  print(("the percetion of prediction is %f "%percentage))
  print("the AUC of prediction is %f"%AUC)
  results_file.write(str(AUC)+'\t'+str(percentage)+'\t')
  print(("Total time comsumed %fs" %(time.time()-start)))
  # Node2Vec END
  # --------------------------------------------------------------------------------- #

  # --------------------------------------------------------------------------------- #
  # Embedding
  # 1. 生成MultiNetsEmbedding需要的东西
  os.system("python3 generate_facts.py %s"%filename)
  # 2. MultiEmbedding
  os.system("./Embedding -network_name %s -generate_flag 0"%filename)


  try:
    mapping_file = 'tmp/node_2_id.txt'
    mapping = {}
    with open(mapping_file,'r+') as f:
      for line in f.readlines():
            node,id = line.strip().split()
            #mapping.append((node,int(id)))
            mapping[node]=int(id)
  except:
    os.system("python3 generate_facts.py %s"%origin_filename)
    mapping_file = 'tmp/node_2_id.txt'
    mapping = {}
    with open(mapping_file,'r+') as f:
      for line in f.readlines():
            node,id = line.strip().split()
            #mapping.append((node,int(id)))
            mapping[node]=int(id)


  matrix_file = 'tmp/%s.node'%filename
  matrix = []
  with open(matrix_file,'r+') as f:
    for line in f.readlines():
          tmp = line.strip().split()
          tmp_float = [float(_) for _ in tmp]
          matrix.append(np.array(tmp_float))
  # Embedding END
  # --------------------------------------------------------------------------------- #


  percentage, AUC = T.run_test(Removelist, matrix, mapping, BFSlist)
  print(("the percetion of prediction is %f "%percentage))
  print("the AUC of prediction is %f"%AUC)
  results_file.write(str(AUC)+'\t'+str(percentage)+'\n')
  print(("Total time comsumed %fs" %(time.time()-start)))

print('ALL DOWN!!!')
