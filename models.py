import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))) 

class MultiHeadAttention(nn.Module):
    """ multi-head attention module """

    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        assert config.d_model % config.nheads == 0, 'Invalid dimensions.'

        self.query_projection = nn.Linear(config.d_model, config.d_model)
        self.key_projection = nn.Linear(config.d_model, config.d_model)
        self.value_projection = nn.Linear(config.d_model, config.d_model)
        self.out_projection = nn.Linear(config.d_model, config.d_model)

        self.nheads = config.nheads
        self.d_model = config.d_model

    def forward(self, z : torch.Tensor, x : torch.Tensor,  mask : torch.Tensor = None):
        batch_size = z.shape[0]
        query_len, key_len, value_len = x.shape[1], z.shape[1], z.shape[1]

        queries = self.query_projection(x)  
        keys, values = self.key_projection(z), self.value_projection(z)

        queries = torch.reshape(queries, (batch_size, query_len, self.nheads, self.d_model // self.nheads))    
        keys = torch.reshape(keys, (batch_size, key_len, self.nheads, self.d_model // self.nheads))            
        values = torch.reshape(values, (batch_size, value_len, self.nheads, self.d_model // self.nheads))     

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            assert z.shape[1] == x.shape[1], "length of sequences must be same."
            energy = energy.masked_fill(mask == 0, float("-inf"))

        attention_scores = torch.softmax(energy / (math.sqrt(self.d_model)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention_scores, values])
        out = out.reshape(batch_size, query_len, self.d_model)
        out = self.out_projection(out)
        return out, attention_scores

class EncoderBlock(nn.Module):
    """ single transformer pre-norm encoder block """

    def __init__(self, config):
        super(EncoderBlock, self).__init__()

        self.d_model = config.d_model
        self.nheads = config.nheads

        self.self_attn = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.layernorm2 = nn.LayerNorm(config.d_model)

        self.mlp = nn.ModuleDict(dict(
            ffnn1 = nn.Linear(config.d_model, 4 * config.d_model),
            act = GELU(),
            ffnn2 = nn.Linear(4 * config.d_model, config.d_model),
            drop = nn.Dropout(config.mlp_drop)
        ))

        m = self.mlp
        self.mlpf = lambda z : m.drop(m.ffnn2(m.act(m.ffnn1(z))))

    def forward(self, z: torch.Tensor):
        z = self.layernorm1(z + self.self_attn(z,z)[0])
        z = self.layernorm2(z + self.mlpf(z))
        return z

class DecoderBlock(nn.Module):
    """ single transformer post-norm decoder block """

    def __init__(self, config):
        super(DecoderBlock, self).__init__()

        self.d_model = config.d_model
        self.nheads = config.nheads

        self.masked_self_attn = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config.d_model)
        self.cross_attn = MultiHeadAttention(config)
        self.layernorm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.ModuleDict(dict(
            ffnn1 = nn.Linear(config.d_model, 4 * config.d_model),
            act = GELU(),
            ffnn2 = nn.Linear(4 * config.d_model, config.d_model),
            drop = nn.Dropout(config.mlp_drop)
        ))

        m = self.mlp
        self.mlpf = lambda x : m.drop(m.ffnn2(m.act(m.ffnn1(x))))

    def forward(self, enc_out: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        x = self.layernorm1(x + self.masked_self_attn(x,x,mask)[0])
        out = self.cross_attn(enc_out, x)[0]
        x = self.layernorm2(x + out)
        x = self.layernorm2(x + self.mlpf(x))
        return x

