import pickle
import bcolz
import numpy as np

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir = 'glove/6B.300.dat', mode = 'w')

glove_file = open('glove/glove.6B.300d.txt','rb')
for l in glove_file:
    line = l.decode().split()
    word = line[0]
    words.append(word)
    word2idx[word] = idx
    idx += 1
    vec = np.array(line[1:]).astype(np.float32)
    vectors.append(vec)

vectors = bcolz.carray(vectors[1:].reshape((400000,300)), rootdir = 'glove/6B.300.dat', mode = 'w')
print(repr(vectors))
vectors.flush()
pickle.dump(words, open('glove/6B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open('glove/6B.300_idx.pkl', 'wb'))
