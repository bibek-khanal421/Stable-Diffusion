import torch 
from torch import nn 
from torch.nn import functional as F 
import math 

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        intermed_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, dim*3) -> split into 3 tensor of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (batch_size, seq_len, dim) -> (batch_size, seq_len, H, dim/H) -> (batch_size, H, seq_len, dim/H)
        q = q.view(intermed_shape).transpose(1, 2)
        k = k.view(intermed_shape).transpose(1, 2)
        v = v.view(intermed_shape).transpose(1, 2)

        # (batch_size, H, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # mask where the upper triangle (above the principal diagonal) is made up of 1 
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)

        # (batch_size, H, seq_len, seq_len) @ (batch_size, H, seq_len, dim/h) -> (batch_size, H, seq_len, dim/H)
        output = weight @ v
        # (batch_size, H, seq_len, dim/H) -> (batch_size, seq_len, H, dim/H)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output


class CrossAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias = in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias = in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent): (batch, seq_len_Q, dim_Q)
        # y: (context): (batch, seq_len_KV, dim_KV) = (batch, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # multiply query by wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)

        output = self.out_proj(output)

        return output 
    


    
        