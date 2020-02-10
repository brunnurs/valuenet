import torch
import torch.nn as nn
import random

vocab_size = 7
embedding_dim_2 = 3

embedding_1 = nn.Embedding(8, 2, padding_idx=0)
embedding_2 = nn.Embedding(8, 5, padding_idx=0)

# Random vector of length 15 consisting of indices 0, ..., 9
x = torch.LongTensor([random.randint(0, 7) for _ in range(15)])
# Adding batch dimension
x = x[None, :]

emb_1 = embedding_1(x)
print(emb_1)
emb_2 = embedding_2(x)
print(emb_2)
# Concatenating embeddings along dimension 2
emb = torch.cat([emb_1, emb_2], dim=2)
print(emb)