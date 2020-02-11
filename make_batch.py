import numpy as np
import pickle

with open('ham_array_specs','rb') as f:
    neg = pickle.load(f)
with open('ham_pos_array_specs','rb') as f:
    pos = pickle.load(f)

for i in range(4342):
    x = np.vstack((pos[20*i:20*(i+1)],neg[76*i:76*(i+1)]))
    y = np.vstack((np.ones((20,1)),np.zeros((76,1))))
    shuffle = np.random.rand(x.shape[0]).argsort()
    np.take(x, shuffle, axis = 0, out = x)
    np.take(y, shuffle, axis = 0, out = y)
    x.dump("ham/inputs" + str(i))
    y.dump("ham/labels" + str(i))
