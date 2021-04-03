import torch
import torch.nn as nn
import math


class DisentangledSelfAttention(nn.Module):
    def __init__(self, attention_head_size, num_attention_heads=1, max_relative_positions=5, dropout=0.5):
        super(DisentangledSelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.pos_ebd_size = max_relative_positions
        out_heads = attention_head_size * num_attention_heads

        self.query_proj = nn.Linear(attention_head_size, out_heads)
        self.key_proj = nn.Linear(attention_head_size, out_heads)
        self.value_proj = nn.Linear(attention_head_size, out_heads)

        self.pos_query_proj = nn.Linear(attention_head_size, out_heads)
        self.pos_key_proj = nn.Linear(attention_head_size, out_heads)

        self.share_att_key = False

    def transpose_for_scores(self, x):
        attention_heads = self.num_attention_heads
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings):
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim() != 4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings))\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings))\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings))\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings))\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)

        # content->position
        c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
        c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))

        # position->content
        r_pos = relative_pos

        p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
        if query_layer.size(-2) != key_layer.size(-2):
            pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
        p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
        if query_layer.size(-2) != key_layer.size(-2):
            p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))

        return c2p_att + p2c_att

    def forward(self, H, pos_embeddings, relative_pos):
        '''
        Shapes:
        H: [batch x seqlen x numfeats]
        pos_embeddings: [2 * maxreldist x numfeats]
        relative_pos: [batch x seqlen x seqlen]
        '''

        key_layer = self.transpose_for_scores(self.key_proj(H))
        query_layer = self.transpose_for_scores(self.query_proj(H))
        value_layer = self.transpose_for_scores(self.value_proj(H))

        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))
        attention_scores += self.disentangled_attention_bias(query_layer, key_layer, relative_pos, pos_embeddings)

        attention_scores /= math.sqrt(query_layer.size(-1) * 3)

        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2),
                                                 attention_scores.size(-1))

        attention_probs = torch.softmax(attention_scores, -1)

        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)),
                                  value_layer)

        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2),
                                           context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
