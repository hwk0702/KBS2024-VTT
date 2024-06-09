import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
import math
from math import sqrt
import numpy as np


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), weights


class VariableAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape
        x = rearrange(x, 'b t f d -> (b t) f d')
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = rearrange(out, '(b t) f d -> b t f d', t=t)
        return out, weights


class TemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape
        x = rearrange(x, 'b t f d -> (b f) t d')
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        if use_attn:
            weights = attn
        else:
            weights = None
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = rearrange(out, '(b f) t d -> b t f d', f=f)
        return out, weights


class VariableTemporalAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.variable_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.variable_to_out = nn.Linear(inner_dim, dim)
        self.variable_dropout = nn.Dropout(dropout)

        self.temporal_to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.temporal_to_out = nn.Linear(inner_dim, dim)
        self.temporal_dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, x, use_attn):
        b, t, f, d = x.shape

        vx = rearrange(x, 'b t f d -> (b t) f d')
        h = self.heads
        vq, vk, vv = self.variable_to_qkv(vx).chunk(3, dim=-1)
        vq, vk, vv = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (vq, vk, vv))
        sim = einsum('b h i d, b h j d -> b h i j', vq, vk) * self.scale

        vattn = sim.softmax(dim=-1)
        if use_attn:
            vweights = vattn
        else:
            vweights = None
        vattn = self.variable_dropout(vattn)

        vout = einsum('b h i j, b h j d -> b h i d', vattn, vv)
        vout = rearrange(vout, 'b h n d -> b n (h d)', h=h)
        vout = self.variable_to_out(vout)
        vout = rearrange(vout, '(b t) f d -> b t f d', t=t)

        tx = rearrange(x, 'b t f d -> (b f) t d')
        tq, tk, tv = self.temporal_to_qkv(tx).chunk(3, dim=-1)
        tq, tk, tv = map(lambda m: rearrange(m, 'b n (h d) -> b h n d', h=h), (tq, tk, tv))
        sim = einsum('b h i d, b h j d -> b h i j', tq, tk) * self.scale

        tattn = sim.softmax(dim=-1)
        if use_attn:
            tweights = tattn
        else:
            tweights = None
        tattn = self.temporal_dropout(tattn)

        tout = einsum('b h i j, b h j d -> b h i d', tattn, tv)
        tout = rearrange(tout, 'b h n d -> b n (h d)', h=h)
        tout = self.temporal_to_out(tout)
        tout = rearrange(tout, '(b f) t d -> b t f d', f=f)

        out = torch.cat([vout, tout], dim=-1)
        out = self.linear(out)
        return out, vweights, tweights