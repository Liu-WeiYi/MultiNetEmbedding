import sys
from link_pred import *
from linking_test import *

# Read the data into a list of strings.

filename = sys.argv[1]
BFSlist, Edgelist = read_data(filename)



T = Test(BFSlist, Edgelist, 0.1)
Removelist, New_BFSlist, New_Edgelist = T.sample()
print('edge list size', len(Edgelist))
print('New edge list size', len(New_Edgelist))

p = Prediction()
vetex_set = p.create_vertex(New_Edgelist)
matrix_old = p.create_adjmatrix(Edgelist, vetex_set)
matrix_new = p.create_adjmatrix(New_Edgelist, vetex_set)
## doing common Neighbors()
cn = CommonNeighbors()
score_cn = cn.fit(matrix_old)
auc_cn = p.auc_score(score_cn, matrix_old, matrix_new, 'cc')

#print(matrix_new)
#print(matrix_old)
#print(score_cn)
print('common neighbors auc', auc_cn)

## doing jaccard
ja = Jaccard()
score_ja = ja.fit(matrix_old)
auc_ja = p.auc_score(score_ja, matrix_old, matrix_new, 'cc')
print('jaccard auc', auc_ja)