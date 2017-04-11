# Sailung Yeung
# <yeungsl@bu.edu>

import numpy as np
import Word2Vec
import Node2Vec
import random, collections, itertools

def read_data(filename):
  """Extract the file and convert it into a neighbor list"""
  BFSlist = {}
  Edgelist = []
  for line in open(filename):
    s, d = line.split(" ")
    src = s
    dst = d
    Edgelist.append((src, dst))
    BFSlist = BFS(src, dst, BFSlist)

  return BFSlist, Edgelist

def BFS(src, dst, BFSlist):
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
  return BFSlist

class Test():
  def __init__(self, BFSlist, Edgelist, p):
    self.E = Edgelist
    self.B = BFSlist
    self.p = p

  def sample(self):
    Edgelist = self.E
    BFSlist = self.B
    p = self.p
    Removelist = []
    New_Edgelist = []
    New_BFSlist = {}

    for edge in Edgelist:
      r = random.random()
      if r < p:
        if len(BFSlist[edge[0]]) == 1 or len(BFSlist[edge[1]]) == 1:
          New_Edgelist.append(edge)
          New_BFSlist = BFS(edge[0], edge[1], New_BFSlist)
          continue
        Removelist.append(edge)
      else:
        New_Edgelist.append(edge)
        New_BFSlist = BFS(edge[0], edge[1], New_BFSlist)
    return Removelist, New_BFSlist, New_Edgelist

  def check(self, src, dst):
    mapping = self.MP
    matrix = self.M
    if type(src) == int and type(dst) == int:
      r = np.linalg.norm(matrix[mapping[str(src)]] - matrix[mapping[str(dst)]])
    elif type(src) == str and type(dst) == str:
      r = np.linalg.norm(matrix[mapping[src.strip()]] - matrix[mapping[dst.strip()]])
    return r

  def run_test(self, Removelist, matrix, mapping, Original_list):
    self.M = matrix
    self.MP = mapping

    p = self.p
    result = {}
    s = 0
    node_list = set([node for edge in Removelist for node in edge])
    #print (len(node_list))
    #print (Removelist)
    #print (mapping)

    for edge in itertools.combinations(node_list, 2):
      dist = self.check(edge[0], edge[1])
      s += dist
      result[edge] = dist

    normalized_result = {}
    for e in result:
      normalized_result[e] = result[e]/s

    #print ("result size", len(normalized_result))

    sorted_result = collections.OrderedDict(sorted(list(normalized_result.items()), key=lambda t:t[1]))

    n = len(normalized_result) * p
    correct = 0
    count = 1
    print(("getting the first ", n))

    for edge in sorted_result:

      if edge[1] in list(Original_list[edge[0]].keys()) or edge[0] in list(Original_list[edge[0]].keys()):

        correct += 1
      count += 1
      if count > n:
        break
    print(('correct', correct))

    AUC = (correct*0.5 + (n-correct))/n
    return correct/n, AUC







