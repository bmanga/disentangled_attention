from attention import DisentangledSelfAttention
import torch
import torch.nn as nn
import numpy as np
from transformer import CustomTransformer
from model import Model
from torchvision.datasets import MNIST


from torch.nn import TransformerEncoderLayer

def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = np.sign(relative_pos)
    mid = bucket_size//2
    abs_pos = np.where((relative_pos<mid) & (relative_pos > -mid), mid-1, np.abs(relative_pos))
    log_pos = np.ceil(np.log(abs_pos/mid)/np.log((max_position-1)/mid) * (mid-1)) + mid
    bucket_pos = np.where(abs_pos<=mid, relative_pos, log_pos*sign).astype(np.int)
    return bucket_pos


def build_relative_positions(query_size, key_size, bucket_size=-1, max_position=-1):
    q_ids = np.arange(0, query_size)
    k_ids = np.arange(0, key_size)
    rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0], 1))
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids

'''

k = 5
N = 5
d = 5

t = DisentangledSelfAttention(d)

H = torch.rand(1, N, d)
pred = torch.rand(1, N, d)
P = nn.Embedding(10, 5).weight
relative_pos = build_relative_positions(N, d)
print(relative_pos)
print(relative_pos.shape)
Ho = t.forward(H, P, relative_pos)
print('*' * 50)
print(H.shape)
print(P.shape)
print(Ho.shape)
'''
'''
N = 5 # length of the input sequence
d = 5 # dimension of the hidden states
k = 5 # max relative distance
t = CustomTransformer(d, nhead=2)

H = torch.rand(1, N, d)
pred = torch.rand(1, N,d )
P = nn.Embedding(10, 5).weight
relative_pos = build_relative_positions(N, d)
Ho = t.forward(H, P, relative_pos)
print(Ho.shape)
'''
d = 5
k = 28 * 28
N = 15
model = Model(num_feats=d, num_classes=10, max_relative_distance=k)
h = torch.rand(1, N, d)
ho = model(h)
print(ho.shape)




