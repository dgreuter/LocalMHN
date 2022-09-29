# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from dgl.nn.pytorch import NNConv
from dgllife.model import MPNNGNN
from dgl.nn.pytorch.utils import Sequential
from dgl.nn.pytorch.glob import MaxPooling

import logging
logger = logging.getLogger()


class MPNN_wo_GRU(nn.Module):
    """MPNN.

    MPNN is introduced in `Neural Message Passing for Quantum Chemistry
    <https://arxiv.org/abs/1704.01212>`__.

    This class performs message passing in MPNN and returns the updated node representations.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node representations. Default to 64.
    edge_in_feats : int
        Size for the input edge features. Default to 128.
    edge_hidden_feats : int
        Size for the hidden edge representations.
    num_step_message_passing : int
        Number of message passing steps. Default to 6.
    """
    def __init__(self, node_in_feats, edge_in_feats, node_out_feats=64,
                 edge_hidden_feats=128, num_step_message_passing=6):
        super().__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_out_feats),
            nn.ReLU()
        )
        self.num_step_message_passing = num_step_message_passing
        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, edge_hidden_feats),
            nn.ReLU(),
            nn.Linear(edge_hidden_feats, node_out_feats * node_out_feats)
        )
        self.gnn_layer = NNConv(
            in_feats=node_out_feats,
            out_feats=node_out_feats,
            edge_func=edge_network,
            aggregator_type='sum'
        )

    def reset_parameters(self):
        """Reinitialize model parameters."""
        self.project_node_feats[0].reset_parameters()
        self.gnn_layer.reset_parameters()
        for layer in self.gnn_layer.edge_func:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()


    def forward(self, g, node_feats, edge_feats):
        """Performs message passing and updates node representations.

        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.

        Returns
        -------
        node_feats : float32 tensor of shape (V, node_out_feats)
            Output node representations.
        """
        node_feats = self.project_node_feats(node_feats) # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)           # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats = node_feats.squeeze(0)

        return node_feats


def pair_atom_feats(g, node_feats):
    sg = g.remove_self_loop() # in case g includes self-loop
    atom_pair_list = torch.transpose(sg.adjacency_matrix().coalesce().indices(), 0, 1)
    atom_pair_idx1 = atom_pair_list[:,0]
    atom_pair_idx2 = atom_pair_list[:,1]
    atom_pair_feats = torch.cat((node_feats[atom_pair_idx1], node_feats[atom_pair_idx2]), dim = 1)
    return atom_pair_feats


class LocalRetro(nn.Module):
    def __init__(self, args, pool=False):
        super().__init__()
        node_size = args.atom_featurizer.feat_size()
        edge_size = args.bond_featurizer.feat_size()
        if pool:
            self.pool = MaxPooling().to(args.device)
        else:
            self.pool = None
        if args.encoder_type in ['MPNN', 'Baseline', 'linear_fps']:
            self.mpnn = MPNNGNN(node_in_feats=node_size,
                               node_out_feats=args.out_size,
                               edge_in_feats=edge_size,
                               edge_hidden_feats=args.hidden_size,
                               num_step_message_passing=args.n_passing).to(args.device)
        elif args.encoder_type == 'MPNN_wo_GRU':
            self.mpnn = MPNN_wo_GRU(node_in_feats=node_size,
                               node_out_feats=args.out_size,
                               edge_in_feats=edge_size,
                               edge_hidden_feats=args.hidden_size,
                               num_step_message_passing=args.n_passing).to(args.device)
        self.linearB = nn.Linear(args.out_size*2, args.out_size).to(args.device)
        self.norm = nn.LayerNorm(args.out_size)
        if args.encoder_type == 'Baseline':
            self.out_layer = nn.Sequential(
                            nn.Linear(args.out_size, args.out_size), 
                            nn.ReLU(), 
                            nn.Dropout(0.2),
                            nn.Linear(args.out_size, args.n_temps)).to(args.device)
        else:
            self.out_layer = None


    def forward(self, g, node_feats, edge_feats):
        node_feats = self.mpnn(g, node_feats, edge_feats)
        atom_feats = node_feats
        bond_feats = self.linearB(pair_atom_feats(g, node_feats))
        if self.pool:
            g.ndata['h'] = atom_feats
            atom_feats = self.pool(g, atom_feats) # n_temps x args.out_size
            g.edata['e'] = bond_feats
            bond_feats = dgl.readout_edges(g, 'e')
        if self.out_layer:
            atom_feats = self.out_layer(atom_feats) 
            bond_feats = self.out_layer(bond_feats) 
        else:
            atom_feats = self.norm(atom_feats)
            bond_feats = self.norm(bond_feats)
        return atom_feats, bond_feats
