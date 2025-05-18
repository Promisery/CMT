import torch
from torch import nn
from torch.nn import functional as F
from typing import List
from torch.nn.init import trunc_normal_, kaiming_normal_
from einops import repeat
from models.x_transformers import AttentionLayers, AbsolutePositionalEmbedding,\
    FixedPositionalEmbedding, l2norm, ReluSquared, GLU, deepnorm_init
from utils import device

TRANSFORMER_DEFAULT_PARAMS = dict(
    use_scalenorm = False,
    use_rmsnorm = True,
    alibi_pos_bias = False,
    alibi_num_heads = None,
    alibi_learned = False,
    rel_pos_bias = False,
    rel_pos_num_buckets = 32,
    rel_pos_max_distance = 128,
    dynamic_pos_bias = False,
    dynamic_pos_bias_log_distance = False,
    dynamic_pos_bias_mlp_depth = 2,
    dynamic_pos_bias_norm = False,
    position_infused_attn = False,
    rotary_pos_emb = False,
    rotary_emb_dim = None,
    custom_layers = None,
    sandwich_coef = None,
    par_ratio = None,
    residual_attn = False,
    cross_residual_attn = False,
    macaron = False,
    pre_norm = True,
    gate_residual = False,
    scale_residual = False,
    scale_residual_constant = 1.,
    drop_path = 0.1,
    deepnorm = False,
    shift_tokens = 0,
    sandwich_norm = True,
    zero_init_branch_output = False,
    ff_mult = 4,
    ff_glu = True,
    ff_relu = False,
    ff_swish = False,
    ff_relu_squared = False,
    ff_post_act_ln = False,
    ff_dropout = 0.1,
    ff_no_bias = False,
    ff_zero_init_output = False,
    attn_talking_heads = False,
    attn_head_scale = False,
    attn_sparse_topk = None,
    attn_use_entmax15 = False,
    attn_num_mem_kv = 0,
    attn_dropout = 0.1,
    attn_on_attn = False,
    attn_gate_values = False,
    attn_zero_init_output = False,
    attn_max_attend_past = None,
    attn_qk_norm = False,
    attn_qk_norm_groups = 1,
    attn_qk_norm_scale = 1,
    attn_one_kv_head = False,
    attn_shared_kv = False,
    attn_value_dim_head = None,
    attn_tensor_product = False,
)


def _get_activation(act, **kwargs):
    if act == 'relu' or act == 'reglu':
        return nn.ReLU()
    elif act == 'gelu' or act == 'geglu':
        return nn.GELU()
    elif act == 'leakyrelu':
        return nn.LeakyReLU()
    elif act == 'silu' or act == 'siglu':
        return nn.SiLU()
    elif act == 'sigmoid' or act == 'glu':
        return nn.Sigmoid()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'softmax':
        return nn.Softmax()
    elif act == 'relu_squared':
        return ReluSquared()
    else:
        raise ValueError(F'Activation {act} not supported!')


class MLP(nn.Module):
    '''
    MLP
    '''
    def __init__(self, in_features, hidden_features=[], out_features=None, act='relu', dropout=0., last_act=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = [hidden_features] if isinstance(hidden_features, int) else hidden_features
        self.shortcut = in_features == out_features
        in_channels = [in_features] + hidden_features
        out_channels = hidden_features + [out_features]

        modules = []
        for idx, (i, o) in enumerate(zip(in_channels, out_channels)):
            if (idx < len(in_channels) - 1) or last_act:
                if 'glu' in act:
                    modules.append(
                        GLU(i, o, activation=_get_activation(act))
                    )
                else:
                    modules.append(nn.Linear(i, o))
                    modules.append(
                        _get_activation(
                            act,
                            dim_in=o,
                            dim_out=o,
                            activation=_get_activation('gelu')
                        )
                    )
                modules.append(nn.Dropout(dropout))
            else:
                modules.append(nn.Linear(i, o))
        
        self.net = nn.Sequential(*modules)

        self._reset_parameters()

    def forward(self, x):
        out = self.net(x)
        return out if not self.shortcut else x + out
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.01)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            

class TransformerEncoder(nn.Module):
    '''
    Transformer encoder
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int,
        num_layers: int,
        pooling: str,
        pe: str,
    ) -> None:
        super().__init__()
        
        assert pooling in ['mean', 'max', 'first', 'last', 'cls', 'attn', 'none'], F'Pooling strategy "{pooling}" not supported!'
        assert set(pe.split('-')).issubset(['rel', 'dyn', 'rot', 'alibi', 'learned', 'fixed', 'none']), F'Positional encoding "{pooling}" not supported!'
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead
        self.num_layers = num_layers
        self.pooling = pooling
        
        if pooling == 'attn':
            self.pooling_attn = nn.Linear(out_channels, 1)
        elif pooling == 'cls':
            self.cls = nn.Parameter(torch.randn(1, 1, out_channels, device=device))
        
        if 'learned' in pe:
            self.pe = AbsolutePositionalEmbedding(out_channels, max_seq_len=100, l2norm_embed=False)
        elif 'fixed' in pe:
            self.pe = FixedPositionalEmbedding(out_channels)
        else:
            self.pe = None
        
        self.emb_layer = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        transformer_update_params = dict(
            rel_pos_bias = ('rel' in pe),
            dynamic_pos_bias = ('dyn' in pe),
            rotary_pos_emb = ('rot' in pe),
            alibi_pos_bias = ('alibi' in pe),
            scale_residual_constant = 0.81 * ((num_layers ** 4) * num_layers) ** .0625 if TRANSFORMER_DEFAULT_PARAMS['deepnorm'] else 1.
        )
        self.transformer_params = TRANSFORMER_DEFAULT_PARAMS.copy()
        self.transformer_params.update(transformer_update_params)
        
        self.encoder = AttentionLayers(
            dim = out_channels,
            depth = num_layers,
            heads = nhead,
            causal = False,
            cross_attend = False,
            only_cross = False,
            **self.transformer_params
        )
                
        self._reset_parameters()
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """Forward pass of transformer encoder

        Args:
            x (Tensor): Input sequence of shape (batch_size, seq_len, in_channels)
            attn_mask (Tensor, optional): Self attention mask. True for attend and False for no attention. Defaults to None.
            key_padding_mask (Tensor, optional): Key padding mask. True for attend and False for no attention. Defaults to None.
        """
        b, t, _ = x.shape
        x = self.emb_layer(x)
        x = x if self.pe is None else x + self.pe(x)
        if self.pooling == 'cls':
            x = torch.cat([repeat(self.cls, '1 1 d -> b t d', b=b, t=t), x], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([torch.ones(x.shape[0], 1, device=x.device, dtype=torch.bool), key_padding_mask], dim=1)
        
        x = self.encoder(x, mask=key_padding_mask, attn_mask=attn_mask)
                
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-1)
            if self.pooling == 'attn':
                x = torch.masked_fill(x, ~key_padding_mask, 0)
                attn = self.pooling_attn(x)
                attn = F.softmax(attn + torch.log((key_padding_mask).float()), dim=1)
                x = torch.sum(x * attn, dim=1)
            elif self.pooling == 'mean':
                x = torch.masked_fill(x, ~key_padding_mask, 0)
                x = torch.sum(x, dim=1) / torch.sum(key_padding_mask, dim=1)
            elif self.pooling == 'max':
                x = torch.masked_fill(x, ~key_padding_mask, -1e9)
                x = torch.max(x, dim=1)[0]
            elif self.pooling == 'first' or self.pooling == 'cls':
                x = x[:, 0]
            elif self.pooling == 'last':
                x = x[:, -1]
        else:
            if self.pooling == 'attn':
                attn = self.pooling_attn(x)
                attn = F.softmax(attn, dim=1)
                x = torch.sum(x * attn, dim=1)   
            elif self.pooling == 'mean':
                x = torch.mean(x, dim=1)
            elif self.pooling == 'max':
                x = torch.max(x, dim=1)
            elif self.pooling == 'first' or self.pooling == 'cls':
                x = x[:, 0]
            elif self.pooling == 'last':
                x = x[:, -1]

        return x
    
    def _reset_parameters(self):
        for p in self.emb_layer.parameters():
            if p.dim() > 1:
                kaiming_normal_(p, nonlinearity='relu', mode='fan_out')
        if self.transformer_params['deepnorm']:
            deepnorm_init(self.encoder, 0.87 * ((self.num_layers ** 4) * self.num_layers) ** -0.0625)
        else:
            for m in self.encoder.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.01)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
                    

class TransformerDecoder(nn.Module):
    '''
    Transformer decoder
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        nhead: int,
        num_layers: int,
        pooling: str,
        pe: str,
        only_cross: bool=False,
    ) -> None:
        super().__init__()
        
        assert pooling in ['mean', 'max', 'first', 'last', 'cls', 'attn', 'none'], F'Pooling strategy "{pooling}" not supported!'
        assert set(pe.split('-')).issubset(['rel', 'dyn', 'rot', 'alibi', 'learned', 'fixed', 'none']), F'Positional encoding "{pooling}" not supported!'
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead
        self.num_layers = num_layers
        self.pooling = pooling
        
        if pooling == 'attn':
            self.pooling_attn = nn.Linear(out_channels, 1)
        elif pooling == 'cls':
            self.cls = nn.Parameter(torch.randn(1, 1, out_channels, device=device))
        
        if 'learned' in pe:
            self.pe = AbsolutePositionalEmbedding(out_channels, max_seq_len=100, l2norm_embed=False)
        elif 'fixed' in pe:
            self.pe = FixedPositionalEmbedding(out_channels)
        else:
            self.pe = None
        
        self.emb_layer = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.mem_emb_layer = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        transformer_update_params = dict(
            rel_pos_bias = ('rel' in pe),
            dynamic_pos_bias = ('dyn' in pe),
            rotary_pos_emb = ('rot' in pe),
            alibi_pos_bias = ('alibi' in pe),
            scale_residual_constant = (3 * num_layers) ** 0.25 if TRANSFORMER_DEFAULT_PARAMS['deepnorm'] else 1.
        )
        self.transformer_params = TRANSFORMER_DEFAULT_PARAMS.copy()
        self.transformer_params.update(transformer_update_params)
        
        self.decoder = AttentionLayers(
            dim = out_channels,
            depth = num_layers,
            heads = nhead,
            causal = False,
            cross_attend = True,
            only_cross = only_cross,
            **self.transformer_params
        )
        
        self._reset_parameters()
    
    def forward(self, x, memory, attn_mask=None, cross_mask=None, key_padding_mask=None, memory_key_padding_mask=None):
        """Forward pass of transformer encoder

        Args:
            x (Tensor): Input sequence of shape (batch_size, seq_len, in_channels)
            attn_mask (Tensor, optional): Self attention mask. True for attend and False for no attention. Defaults to None.
            key_padding_mask (Tensor, optional): Key padding mask. True for attend and False for no attention. Defaults to None.
        """
        b, t, _ = x.shape
        x = self.emb_layer(x)
        x = x if self.pe is None else x + self.pe(x)
        if self.pooling == 'cls':
            x = torch.cat([repeat(self.cls, '1 1 d -> b t d', b=b, t=t), x], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([torch.ones(x.shape[0], 1, device=x.device, dtype=torch.bool), key_padding_mask], dim=1)
        
        x = self.decoder(x, memory, mask=key_padding_mask, context_mask=memory_key_padding_mask, attn_mask=attn_mask, cross_mask=cross_mask)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(-1)
            if self.pooling == 'attn':
                x = torch.masked_fill(x, ~key_padding_mask, 0)
                attn = self.pooling_attn(x)
                attn = F.softmax(attn + torch.log((key_padding_mask).float()), dim=1)
                x = torch.sum(x * attn, dim=1)
            elif self.pooling == 'mean':
                x = torch.masked_fill(x, ~key_padding_mask, 0)
                x = torch.sum(x, dim=1) / torch.sum(key_padding_mask, dim=1)
            elif self.pooling == 'max':
                x = torch.masked_fill(x, ~key_padding_mask, -1e9)
                x = torch.max(x, dim=1)[0]
            elif self.pooling == 'first' or self.pooling == 'cls':
                x = x[:, 0]
            elif self.pooling == 'last':
                x = x[:, -1]
        else:
            if self.pooling == 'attn':
                attn = self.pooling_attn(x)
                attn = F.softmax(attn, dim=1)
                x = torch.sum(x * attn, dim=1)   
            elif self.pooling == 'mean':
                x = torch.mean(x, dim=1)
            elif self.pooling == 'max':
                x = torch.max(x, dim=1)
            elif self.pooling == 'first' or self.pooling == 'cls':
                x = x[:, 0]
            elif self.pooling == 'last':
                x = x[:, -1]

        return x
    
    def _reset_parameters(self):
        for p in self.emb_layer.parameters():
            if p.dim() > 1:
                kaiming_normal_(p, nonlinearity='relu', mode='fan_out')
        for p in self.mem_emb_layer.parameters():
            if p.dim() > 1:
                kaiming_normal_(p, nonlinearity='relu', mode='fan_out')
        if self.transformer_params['deepnorm']:
            deepnorm_init(self.decoder, (12 * self.num_layers) ** -0.25)
        else:
            for m in self.decoder.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.01)
                    if isinstance(m, nn.Linear) and m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)


class AgentEmbedding(nn.Module):
    def __init__(self, dim: int, agents: List, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.agents = {agent: idx for idx, agent in enumerate(agents)}
        self.max_seq_len = len(agents)
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(self.max_seq_len, dim)

    def forward(self, x, agent: str):        
        pos = torch.ones(1, device=x.device, dtype=torch.long) * self.agents[agent]

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb


class NOIEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int=100, l2norm_embed: bool = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(self.max_seq_len, dim)

    def forward(self, x: torch.Tensor):        

        pos_emb = self.emb(x)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb