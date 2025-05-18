import torch
import torch.nn as nn
from models.base import TransformerEncoder, MLP, NOIEmbedding
from typing import Dict
from utils import device


class Selector(nn.Module):
    def __init__(self, args: Dict):
        super().__init__()

        self.pi_op = MLP(
            2 * args['node_enc_size'] + args['target_agent_enc_size'] + 2,
            [args['pi_h1_size'], args['pi_h2_size']],
            1,
            args['activation'],
            dropout=0.1
        )
        
        self.pi_op_goal = MLP(
            args['node_enc_size'] + args['target_agent_enc_size'],
            [args['pi_h1_size'], args['pi_h2_size']],
            1,
            args['activation'],
            dropout=0.1
        )
        self.log_softmax = nn.LogSoftmax(dim=2)

        # For sampling policy
        self.horizon = args['horizon']
        self.num_noi = args['num_noi']
        
        # Attention based aggregator
        self.noi_pos_emb = NOIEmbedding(args['node_enc_size'], self.num_noi+1)
        self.noi_encoder = TransformerEncoder(
            args['node_enc_size'] + args['target_agent_enc_size'],
            args['noi_enc_size'],
            args['noi_n_head'],
            args['noi_enc_layers'],
            args['noi_pooling'],
            args['noi_pe'],
        )

    def forward(self, encodings: Dict) -> Dict:
        
        # Unpack encodings:
        target_agent_encoding = encodings['target_agent_encoding']
        node_encodings = encodings['context_encoding']['combined']
        node_masks = encodings['context_encoding']['combined_masks']
        s_next = encodings['s_next']
        edge_type = encodings['edge_type']

        # Compute pi (log probs)
        pi = self.compute_policy(target_agent_encoding, node_encodings, node_masks, s_next, edge_type)

        init_node = encodings['init_node']
        node_prob, node_sequence = self.sample_policy(pi, s_next, init_node)

        # Selectively aggregate context along traversed paths
        agg_enc, noi_mask = self.aggregate(node_prob, node_sequence, node_encodings, target_agent_encoding)

        outputs = encodings.copy()
        outputs.update({'agg_encoding': agg_enc, 'noi_mask': noi_mask, 'pi': pi})
        return outputs

    def aggregate(self, node_prob, node_sequence, node_encodings, target_agent_encoding) -> torch.Tensor:

        noi_prob, noi_idx = torch.topk(node_prob, k=self.num_noi, dim=-1)
        noi_sequence = node_sequence.gather(dim=1, index=noi_idx)
        noi_encodings = node_encodings.gather(dim=1, index=noi_idx.unsqueeze(-1).repeat(1, 1, node_encodings.shape[-1]))

        noi_encodings = noi_encodings + self.noi_pos_emb(noi_sequence)
        noi_mask = noi_sequence.bool()
        
        noi_encodings = torch.cat([noi_encodings, target_agent_encoding.unsqueeze(1).repeat(1, noi_encodings.shape[1], 1)], dim=-1)
        
        agg_enc = self.noi_encoder(noi_encodings, key_padding_mask=noi_mask)

        return agg_enc, noi_mask

    def sample_policy(self, pi, s_next, init_node) -> torch.Tensor:
        
        with torch.no_grad():
            
            # convert log probs to probs
            pi = torch.nan_to_num(torch.exp(pi), 0.)
            
            # Useful variables:
            batch_size = pi.shape[0]
            max_nodes = pi.shape[1]

            # Initialize output
            node_prob = torch.zeros(batch_size, max_nodes, device=device)
            node_prob[init_node.bool()] = 1.
            node_prob_dummy = torch.zeros_like(node_prob)
            node_prob = torch.cat([node_prob, node_prob_dummy], dim=1)
            
            # Set up dummy self transitions for goal states:
            pi_dummy = torch.zeros_like(pi)
            pi_dummy[:, :, -1] = 1
            s_next_dummy = torch.zeros_like(s_next)
            s_next_dummy[:, :, -1] = max_nodes + torch.arange(max_nodes).unsqueeze(0).repeat(batch_size, 1)
            pi = torch.cat((pi, pi_dummy), dim=1)
            s_next = torch.cat((s_next, s_next_dummy), dim=1)
            s_next = s_next.reshape(batch_size, -1).long()
            
            # Sample initial node:
            node_sequence = torch.zeros(*node_prob.shape, device=device, dtype=torch.long)
            node_sequence.masked_fill_(node_prob.bool(), 1)
            prev_node_prob = node_prob.clone()
            max_node_prob = node_prob.clone()
            for n in range(1, self.horizon):
                transfer_prob = (prev_node_prob.unsqueeze(-1) * pi).reshape(batch_size, -1)
                next_node_prob = torch.scatter_add(torch.zeros_like(prev_node_prob), dim=1, index=s_next, src=transfer_prob)
                seq_mask = next_node_prob.bool() & (max_node_prob < next_node_prob)
                node_sequence.masked_fill_(seq_mask, n+1)
                node_prob = node_prob + next_node_prob
                prev_node_prob = next_node_prob
                max_node_prob = torch.max(max_node_prob, next_node_prob)
            
            node_prob = node_prob[:, :max_nodes]
            node_sequence = node_sequence[:, :max_nodes]
                        
        return node_prob, node_sequence

    def compute_policy(self, target_agent_encoding, node_encodings, node_masks, s_next, edge_type) -> torch.Tensor:
        """
        Forward pass for policy header
        :param target_agent_encoding: tensor encoding the target agent's past motion
        :param node_encodings: tensor of node encodings provided by the encoder
        :param node_masks: masks indicating whether a node exists for a given index in the tensor
        :param s_next: look-up table for next node for a given source node and edge
        :param edge_type: look-up table with edge types
        :return pi: tensor with probabilities corresponding to the policy
        """
        # Useful variables:
        batch_size = node_encodings.shape[0]
        max_nodes = node_encodings.shape[1]
        max_nbrs = s_next.shape[2] - 1
        node_enc_size = node_encodings.shape[2]
        target_agent_enc_size = target_agent_encoding.shape[1]

        # Gather source node encodings, destination node encodings, edge encodings and target agent encodings.
        src_node_enc = node_encodings.unsqueeze(2).repeat(1, 1, max_nbrs, 1)
        dst_idcs = s_next[:, :, :-1].reshape(batch_size, -1).long()
        batch_idcs = torch.arange(batch_size).unsqueeze(1).repeat(1, max_nodes * max_nbrs)
        dst_node_enc = node_encodings[batch_idcs, dst_idcs].reshape(batch_size, max_nodes, max_nbrs, node_enc_size)
        target_agent_enc = target_agent_encoding.unsqueeze(1).unsqueeze(2).repeat(1, max_nodes, max_nbrs, 1)
        edge_enc = torch.cat((torch.as_tensor(edge_type[:, :, :-1] == 1, device=device).unsqueeze(3).float(),
                              torch.as_tensor(edge_type[:, :, :-1] == 2, device=device).unsqueeze(3).float()), dim=3)
        enc = torch.cat((target_agent_enc, src_node_enc, dst_node_enc, edge_enc), dim=3)
        enc_goal = torch.cat((target_agent_enc[:, :, 0, :], src_node_enc[:, :, 0, :]), dim=2)

        # Form a single batch of encodings
        masks = torch.sum(edge_enc, dim=3, keepdim=True).bool()
        masks_goal = ~node_masks.unsqueeze(-1).bool()
        enc_batched = torch.masked_select(enc, masks).reshape(-1, target_agent_enc_size + 2*node_enc_size + 2)
        enc_goal_batched = torch.masked_select(enc_goal, masks_goal).reshape(-1, target_agent_enc_size + node_enc_size)

        # Compute scores for pi_route
        pi_ = self.pi_op(enc_batched)
        pi = torch.zeros_like(masks).float()
        pi = pi.masked_scatter_(masks, pi_).squeeze(-1)
        pi_goal_ = self.pi_op_goal(enc_goal_batched)
        pi_goal = torch.zeros_like(masks_goal).float()
        pi_goal = pi_goal.masked_scatter_(masks_goal, pi_goal_)

        # Normalize to give log probabilities
        pi = torch.cat((pi, pi_goal), dim=-1)
        op_masks = torch.log(torch.as_tensor(edge_type != 0).float())
        pi = self.log_softmax(pi + op_masks)

        return pi