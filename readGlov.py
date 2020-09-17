import os
import numpy as np
import torch
import pickle

embeddings_index = {}
with open('glove.6B/glove.6B.50d.txt', 'r') as glove:
    for line in glove:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
glove.close()

word_to_id = {}
embedding_matrix = np.random.random((len(embeddings_index), 50))
for i, item in enumerate(embeddings_index.items()):
    word_to_id[item[0]] = i
    embedding_vector = embeddings_index.get(item[0])
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print(" Completed reading Glov.50d, now starting training.....")

torch.save(embedding_matrix, os.path.join('glove.6B', 'word_embeddings.pth'))
with open(os.path.join('glove.6B', 'wordtoid.pickle'), 'wb') as f:
    pickle.dump(word_to_id, f)
f.close
