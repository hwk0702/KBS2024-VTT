import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Attention import Attention
import math


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class moving_avg1d(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg1d, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class moving_avg2d(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg2d, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1, 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1, 1)
        x = torch.cat([front, x, end], dim=1)
        front = x[:, :, :, 0:1].repeat(1, 1, 1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2))
        end = x[:, :, :, -1:].repeat(1, 1, 1, math.floor((self.kernel_size - 1) // 2))
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x.permute(0, 2, 1, 3))
        x = x.permute(0, 2, 1, 3)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout)),
                PreNorm(dim, FeedForward(dim, dropout=ff_dropout)),
            ]))

    def forward(self, x, use_attn):
        h = x
        attn_weights = []
        for attn, ff in self.layers:
            x, weights = attn(h, use_attn=use_attn)
            h = x + h
            attn_weights.append(weights)
            x = ff(h)
            h = x + h

        return h, attn_weights


class VariableTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x
