import torch
import torch.nn as nn
from models.base import MLP, TransformerDecoder, MLP
from models.decoders.utils import bivariate_gaussian_activation_with_anchors, gmm_anchors
from models.decoders.utils import gmrc
from typing import Dict, Union
from datasets.interface import SingleAgentDataset
from einops import rearrange, repeat


class Decoder(nn.Module):

    def __init__(self, args):
        
        super().__init__()
        self.num_modes = args['num_modes']
        self.k = args['k']
        self.op_len = args['op_len']
        
        self.n_sample = args['n_sample']
        
        self.op_traj = MLP(
            args['anchor_size'],
            args['hidden_size'],
            self.op_len * 5,
            act=args['activation'],
            dropout=0.
        )
        self.op_prob = MLP(
            args['anchor_size'],
            args['hidden_size'],
            1,
            act=args['activation'],
            dropout=0.
        )
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.anchors = nn.Parameter(torch.zeros(self.num_modes, self.op_len * 4), requires_grad=False)
        
        self.anchor_emb = MLP(
            in_features=self.op_len * 4,
            out_features=args['noi_enc_size'],
            act=args['activation'],
            dropout=0.,
            last_act=True,
        )
        
        self.anchor_decoder = TransformerDecoder(
            args['noi_enc_size'],
            args['anchor_size'],
            args['anchor_n_head'],
            args['anchor_layers'],
            args['anchor_pooling'],
            args['anchor_pe'],
            only_cross=False,
        )

    def generate_anchors(self, ds: SingleAgentDataset=None, anchors: torch.Tensor=None):
        """
        Function to initialize anchors
        :param ds: train dataset for single agent trajectory prediction
        """
        if anchors is not None:
            self.anchors = nn.Parameter(anchors.reshape(self.num_modes, -1))
        else:
            assert ds is not None
            self.anchors = nn.Parameter(gmm_anchors(self.num_modes, ds).reshape(self.num_modes, -1))
    
    def forward(self, inputs: Union[Dict, torch.Tensor]) -> Dict:

        agg_encoding = inputs['agg_encoding']
        noi_mask = inputs['noi_mask']
        batch_size = agg_encoding.shape[0]
        
        anchors_emb = repeat(self.anchor_emb(self.anchors), 'n d -> b n d', b=batch_size)

        mode_encodings = self.anchor_decoder(anchors_emb, agg_encoding, memory_key_padding_mask=noi_mask)

        # Output trajectories
        traj = self.op_traj(mode_encodings)
        traj = traj.reshape(batch_size, self.num_modes, self.op_len, 5)
        anchors = rearrange(self.anchors, 'n (t d) -> 1 n t d', t=self.op_len)
        traj = bivariate_gaussian_activation_with_anchors(traj, anchors)
        
        log_probs = self.log_softmax(self.op_prob(mode_encodings).squeeze(-1))
        
        if not self.training and (self.num_modes > self.k):
            traj, probs, _ = gmrc(
                traj,
                log_probs,
                self.k,
                n_sample=self.n_sample,
            )
        else:
            probs = torch.exp(log_probs)
        
        predictions = inputs.copy()
        predictions.update({'traj': traj, 'probs': probs, 'log_probs': log_probs})

        if type(inputs) is dict:
            for key, val in inputs.items():
                if key != 'agg_encoding':
                    predictions[key] = val

        return predictions