import torch
from torch.nn import Module, Linear, LayerNorm, Embedding, BatchNorm1d, BatchNorm2d, BatchNorm3d, GRU
from inspect import isfunction
from collections.abc import Iterable


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def exists(val):
    return val is not None

def default(val, default):
    if exists(val):
        return val
    return default() if isfunction(default) else default

def get_to_n_tuple(n):
    def f(val):
        return val if isinstance(val, Iterable) else (val,) * n
    return f

to_2_tuple = get_to_n_tuple(2)
to_3_tuple = get_to_n_tuple(3)
to_4_tuple = get_to_n_tuple(4)


def get_wd_params(module: Module):
    """Weight decay is only applied to a part of the params.
    https://github.com/karpathy/minGPT   

    Args:
        module (Module): torch.nn.Module

    Returns:
        optim_groups: Separated parameters
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (Linear, GRU)
    blacklist_weight_modules = (LayerNorm, Embedding, BatchNorm1d, BatchNorm2d, BatchNorm3d)
    for mn, m in module.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.startswith('bias'):
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.startswith('weight') and isinstance(m, whitelist_weight_modules):
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('g') or pn.endswith('anchors'):
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in module.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))]},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    return optim_groups