from transformer import CustomTransformer
import torch.nn as nn
import torch
import numpy as np
import math


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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, num_feats, num_classes, max_relative_distance, nhead=8, hidden_mlp_size=1024):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(24, num_feats, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.transformer = CustomTransformer(num_feats, nhead=nhead, num_encoder_layers=2, dim_feedforward=hidden_mlp_size)
        self.lin1 = nn.Linear(num_feats, hidden_mlp_size)
        self.relu3 = nn.ReLU(inplace=True)
        self.pred = nn.Linear(hidden_mlp_size, num_classes)

        self.P = nn.Embedding(2 * max_relative_distance, num_feats).weight
        self.P.requires_grad = False
        self.num_feats = num_feats
        self.relative_pos_cache = None
        self.prev_N = -1

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.flatten(-2).permute(0, 2, 1)
        N = x.size(-2)
        if self.prev_N != N:
            self.relative_pos_cache = build_relative_positions(N, N)
        x = self.transformer(x, self.P, self.relative_pos_cache)
        x = self.lin1(x[:, 0])
        x = self.relu3(x)
        x = self.pred(x)
        return x
