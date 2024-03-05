import torch 
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention
from decouple import config

CLIP_EMBEDDING_SIZE=config('CLIP_EMBEDDING_SIZE')
CLIP_VOCAB_SIZE=config('CLIP_VOCAB_SIZE')
MAX_SEQ_LEN=config('MAX_SEQ_LEN')
CLIP_HEADS=config('CLIP_HEADS')
TOTAL_CLIP_LAYER=config('TOTAL_CLIP_LAYER')


class CLIPEmbedding(nn.Module):
    
    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.positional_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))
    
    def forward(self, tokens: torch.LongTensor):
        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        x = self.token_embedding(tokens)
        x += self.positional_embedding
        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int) -> None:
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, dim)
        residue = x 
        # self attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue
        # feed forward layer 
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        #QuickGELU activation function
        x = x * torch.sigmoid(1.702 * x) 
        x = self.linear_2(x)
        x += residue 
        return x 


class CLIP(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.embedding = CLIPEmbedding(CLIP_VOCAB_SIZE, CLIP_EMBEDDING_SIZE, MAX_SEQ_LEN)

        self.layers = nn.Module([
            CLIPLayer(CLIP_HEADS, CLIP_EMBEDDING_SIZE) for i in range(TOTAL_CLIP_LAYER)
        ])

        self.layernorm = nn.LayerNorm(CLIP_EMBEDDING_SIZE)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (batch_size, seq_len) -> (batch_size, seq_len, dim)
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)
        
        output = self.layernorm(state)

        return output

