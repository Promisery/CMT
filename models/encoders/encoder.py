from models.base import TransformerEncoder, TransformerDecoder, AgentEmbedding, MLP
import torch
import torch.nn as nn
from typing import Dict
from einops import repeat
from einops.layers.torch import Rearrange
from utils import device


class Encoder(nn.Module):

    def __init__(self, args: Dict):
        super().__init__()
        
        assert args['norm'] in ['pre', 'post', 'none'], 'Norm must be pre, post, or none'
        
        self.random_mask_keep_prob = 1 - args['random_mask_prob']
        
        # Target agent encoder
        target_agent_pre_norm = nn.Sequential(
            Rearrange('b t d -> b d t'),
            nn.BatchNorm1d(args['target_agent_feat_size']),
            Rearrange('b d t -> b t d')
        ) if args['norm'] == 'pre' else nn.Identity()
        
        target_agent_post_norm = nn.Sequential(
            Rearrange('b t d -> b d t'),
            nn.BatchNorm1d(args['emb_size']),
            Rearrange('b d t -> b t d')
        ) if args['norm'] == 'post' else nn.Identity()
        
        self.target_agent_emb = nn.Sequential(
            target_agent_pre_norm,
            MLP(
                in_features=args['target_agent_feat_size'],
                out_features=args['emb_size'],
                act=args['activation'],
                dropout=0.,
                last_act=True,
            ),
            target_agent_post_norm,
        )
        
        # Surrounding agent encoder
        nbr_pre_norm = nn.Sequential(
            Rearrange('b n t d -> b d n t'),
            nn.BatchNorm2d(args['nbr_feat_size']),
            Rearrange('b d n t -> b n t d')
        ) if args['norm'] == 'pre' else nn.Identity()
        
        nbr_post_norm = nn.Sequential(
            Rearrange('b n t d -> b d n t'),
            nn.BatchNorm2d(args['emb_size']),
            Rearrange('b d n t -> b n t d')
        ) if args['norm'] == 'post' else nn.Identity()
        
        self.nbr_emb = nn.Sequential(
            nbr_pre_norm,
            MLP(
                in_features=args['nbr_feat_size'],
                out_features=args['emb_size'],
                act=args['activation'],
                dropout=0.,
                last_act=True,
            ),
            nbr_post_norm,
        )
        
        self.agent_emb = AgentEmbedding(
            args['emb_size'],
            agents=['target', 'veh', 'ped'],
        )
        
        self.agent_enc = TransformerEncoder(
            args['emb_size'],
            args['enc_size'],
            args['enc_n_head'],
            args['enc_layers'],
            args['enc_pooling'],
            args['enc_pe'],
        )
        
        self.agent_fusion = TransformerEncoder(
            args['enc_size'],
            args['fusion_size'],
            args['fusion_n_head'],
            args['fusion_layers'],
            args['fusion_pooling'],
            args['fusion_pe'],
        )
        
        # Node encoders
        node_pre_norm = nn.Sequential(
            Rearrange('b n t d -> b d n t'),
            nn.BatchNorm2d(args['node_feat_size']),
            Rearrange('b d n t -> b n t d')
        ) if args['norm'] == 'pre' else nn.Identity()
        
        node_post_norm = nn.Sequential(
            Rearrange('b n t d -> b d n t'),
            nn.BatchNorm2d(args['emb_size']),
            Rearrange('b d n t -> b n t d')
        ) if args['norm'] == 'post' else nn.Identity()
        
        self.node_emb = nn.Sequential(
            node_pre_norm,
            MLP(
                in_features=args['node_feat_size'],
                out_features=args['emb_size'],
                act=args['activation'],
                dropout=0.,
                last_act=True,
            ),
            node_post_norm,
        )
        
        self.node_encoder = TransformerEncoder(
            args['emb_size'],
            args['enc_size'],
            args['enc_n_head'],
            args['enc_layers'],
            args['enc_pooling'],
            args['enc_pe'],
        )

        self.node_fusion = TransformerDecoder(
            args['enc_size'],
            args['dec_size'],
            args['dec_n_head'],
            args['dec_layers'],
            args['dec_pooling'],
            args['dec_pe'],
        )

    def forward(self, inputs: Dict) -> Dict:
        
        # Encode target agent
        target_agent_feats = inputs['target_agent_representation']
        target_agent_embedding = self.target_agent_emb(target_agent_feats)
        target_agent_embedding = target_agent_embedding + self.agent_emb(target_agent_embedding, 'target')

        # Encode lane nodes
        lane_node_feats = inputs['map_representation']['lane_node_feats']
        lane_node_masks = inputs['map_representation']['lane_node_masks']
        lane_node_embedding = self.node_emb(lane_node_feats)
        lane_node_enc = self.variable_size_transformer_encode(lane_node_embedding, lane_node_masks, self.node_encoder, random_mask=False)

        # Encode surrounding agents
        nbr_vehicle_feats = inputs['surrounding_agent_representation']['vehicles']
        nbr_vehicle_masks = inputs['surrounding_agent_representation']['vehicle_masks']
        nbr_vehicle_embedding = self.nbr_emb(nbr_vehicle_feats)
        nbr_vehicle_embedding = nbr_vehicle_embedding + self.agent_emb(nbr_vehicle_embedding, 'veh')
        
        nbr_ped_feats = inputs['surrounding_agent_representation']['pedestrians']
        nbr_ped_masks = inputs['surrounding_agent_representation']['pedestrian_masks']
        nbr_ped_embedding = self.nbr_emb(nbr_ped_feats)
        nbr_ped_embedding = nbr_ped_embedding + self.agent_emb(nbr_ped_embedding, 'ped')
        
        agent_embedding = torch.cat((nbr_vehicle_embedding, nbr_ped_embedding, target_agent_embedding.unsqueeze(1)), dim=1)
        agent_masks = torch.cat((nbr_vehicle_masks, nbr_ped_masks, torch.zeros_like(nbr_vehicle_masks[:, :1])), dim=1)
        
        context_encodings = self.variable_size_transformer_encode(agent_embedding, agent_masks, self.agent_enc, random_mask=self.training)
        agent_masks = ~agent_masks[..., 0].all(dim=-1)
        context_encodings = self.agent_fusion(context_encodings, key_padding_mask=agent_masks)
        target_agent_enc = context_encodings[:, -1]

        attn_veh_mask = inputs['agent_node_masks']['vehicles'].bool()
        attn_ped_mask = inputs['agent_node_masks']['pedestrians'].bool()
        attn_target_mask = torch.zeros(attn_veh_mask.shape[0], attn_veh_mask.shape[1], 1, dtype=torch.bool, device=attn_veh_mask.device)

        context_mask = ~torch.cat((attn_veh_mask, attn_ped_mask, attn_target_mask), dim=2)
        context_mask = repeat(context_mask, 'b n a -> (b h) n a', h=self.node_fusion.nhead)
        
        adj_mat = self.build_adj_mat(inputs['map_representation']['s_next'], inputs['map_representation']['edge_type'])
        adj_mat = repeat(adj_mat, 'b n1 n2 -> (b h) n1 n2', h=self.node_fusion.nhead)
        lane_node_enc = self.node_fusion(lane_node_enc, context_encodings, attn_mask=adj_mat, cross_mask=context_mask)

        # Lane node masks
        lane_node_masks = ~lane_node_masks[:, :, :, 0].bool()
        lane_node_masks = lane_node_masks.any(dim=2)
        lane_node_masks = ~lane_node_masks
        lane_node_masks = lane_node_masks.float()

        # Return encodings
        encodings = inputs.copy()
        encodings.update({
            'target_agent_encoding': target_agent_enc,
            'context_encoding': {
                'combined': lane_node_enc,
                'combined_masks': lane_node_masks,
                'map': None,
                'vehicles': None,
                'pedestrians': None,
                'map_masks': None,
                'vehicle_masks': None,
                'pedestrian_masks': None
            },
        })

        # Pass on initial nodes and edge structure to aggregator if included in inputs
        if 'init_node' in inputs:
            encodings['init_node'] = inputs['init_node']
            encodings['node_seq_gt'] = inputs['node_seq_gt']
            encodings['s_next'] = inputs['map_representation']['s_next']
            encodings['edge_type'] = inputs['map_representation']['edge_type']

        return encodings

    def variable_size_transformer_encode(self, feat_embedding: torch.Tensor, masks: torch.Tensor, tf: TransformerEncoder, random_mask: bool=False) -> torch.Tensor:
        """
        Returns encoding for a batch of inputs where each sample in the batch is a set of a variable number
        of sequences, of variable lengths.
        """

        # Form a large batch of all sequences in the batch
        masks_for_batching = ~masks[:, :, :, 0].bool()
        if random_mask:
            random_mask = torch.bernoulli(torch.empty(masks_for_batching.shape, device=device).fill_(self.random_mask_keep_prob)).bool()
            masks_for_batching = masks_for_batching & random_mask
        masks_for_batching = masks_for_batching.any(dim=-1).unsqueeze(2).unsqueeze(3)
        feat_embedding_batched = torch.masked_select(feat_embedding, masks_for_batching)
        feat_embedding_batched = feat_embedding_batched.view(-1, feat_embedding.shape[2], feat_embedding.shape[3])
        masks_for_batching = masks_for_batching.squeeze(3)
        masks_batched = ~torch.masked_select(masks[..., 0], masks_for_batching).bool()
        masks_batched = masks_batched.view(feat_embedding_batched.shape[0], feat_embedding_batched.shape[1])

        if not masks_batched.all():
            encoding_batched = tf(feat_embedding_batched, key_padding_mask=masks_batched)
            masks_for_scattering = masks_for_batching.repeat(1, 1, encoding_batched.shape[-1])
            encoding = torch.zeros(masks_for_scattering.shape, device=device)
            encoding = encoding.masked_scatter(masks_for_scattering, encoding_batched)

        else:
            batch_size = feat_embedding.shape[0]
            max_num = feat_embedding.shape[1]
            hidden_state_size = tf.out_channels
            encoding = torch.zeros((batch_size, max_num, hidden_state_size), device=device)

        return encoding

    @staticmethod
    def build_adj_mat(s_next, edge_type):
        """
        Builds adjacency matrix for GAT layers.
        """
        batch_size = s_next.shape[0]
        max_nodes = s_next.shape[1]
        max_edges = s_next.shape[2]
        adj_mat = torch.diag(torch.ones(max_nodes, device=device)).unsqueeze(0).repeat(batch_size, 1, 1).bool()

        dummy_vals = torch.arange(max_nodes, device=device).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        dummy_vals = dummy_vals.float()
        s_next[edge_type == 0] = dummy_vals[edge_type == 0]
        batch_indices = torch.arange(batch_size).unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_edges)
        src_indices = torch.arange(max_nodes).unsqueeze(0).unsqueeze(2).repeat(batch_size, 1, max_edges)
        adj_mat[batch_indices[:, :, :-1], src_indices[:, :, :-1], s_next[:, :, :-1].long()] = True
        adj_mat = adj_mat | torch.transpose(adj_mat, 1, 2)

        return adj_mat
