import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.Embed import PositionalEmbedding, CausalConv1d
from layers.Attention import VariableTemporalAttention
from layers.Transformer_Enc import PreNorm


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.feature_num = params['feature_num']
        self.num_transformer_blocks = params['n_layer']
        self.num_heads = params['n_head']
        self.embedding_dims = params['hidden_size']
        self.attn_dropout = params['attn_pdrop']
        self.ff_dropout = params['resid_pdrop']
        self.time_emb = params['time_emb']

        self.position_embedding = PositionalEmbedding(d_model=self.embedding_dims)
        self.value_embedding = nn.Linear(self.time_emb, self.embedding_dims)

        self.causal_conv1 = CausalConv1d(self.feature_num,
                                         self.feature_num,
                                         kernel_size=4,
                                         dilation=1,
                                         groups=self.feature_num)
        self.causal_conv2 = CausalConv1d(self.feature_num,
                                         self.feature_num,
                                         kernel_size=8,
                                         dilation=2,
                                         groups=self.feature_num)
        self.causal_conv3 = CausalConv1d(self.feature_num,
                                         self.feature_num,
                                         kernel_size=16,
                                         dilation=3,
                                         groups=self.feature_num)

        # transformer
        self.transformer_layers = nn.ModuleList([])

        for _ in range(self.num_transformer_blocks):
            self.transformer_layers.append(PreNorm(self.embedding_dims, VariableTemporalAttention(self.embedding_dims,
                                                                                                  heads=self.num_heads,
                                                                                                  dim_head=self.embedding_dims,
                                                                                                  dropout=self.attn_dropout)),
                                           )

        self.dropout = nn.Dropout(self.ff_dropout)

        # self.transformer_layers = nn.Sequential(*self.transformer_layers)
        self.mlp_head = nn.Linear(self.feature_num*self.embedding_dims, self.feature_num)
        # self.mlp_head = PreNorm(self.feature_num*self.embedding_dims, nn.Linear(self.feature_num*self.embedding_dims, self.feature_num))
        # self.mlp_head = nn.Sequential(nn.Linear(self.feature_num*self.embedding_dims, self.feature_num), nn.Sigmoid())


    def forward(self, x, use_attn=False):
        variable_attn_weights = []
        temporal_attn_weights = []
        b, w, f = x.shape

        x = rearrange(x, 'b w f -> b f w')
        conv1 = self.causal_conv1(x)
        conv2 = self.causal_conv2(x)
        conv3 = self.causal_conv3(x)

        x = torch.stack([x, conv1, conv2, conv3], dim=-1)
        x = rearrange(x, 'b f w d -> b w f d')
        x = self.value_embedding(x)

        position_emb = self.position_embedding(x)
        position_emb = repeat(position_emb, 'b t d -> b t f d', f=f)
        x += position_emb
        x = self.dropout(x)
        h = x
        for attn in self.transformer_layers:
            x, vweights, tweights = attn(h, use_attn=use_attn)
            h = x + h
            variable_attn_weights.append(vweights)
            temporal_attn_weights.append(tweights)

        h = rearrange(h, 'b w f d -> b w (f d)')
        h = self.mlp_head(h)
        
        # Convert lists to tensors
        if use_attn:
            variable_attn_weights = torch.stack(variable_attn_weights, dim=0)
            temporal_attn_weights = torch.stack(temporal_attn_weights, dim=0)
        
        return h, [variable_attn_weights, temporal_attn_weights]