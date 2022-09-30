# -*- coding: utf-8 -*-
"""
Author:     Dominik Greuter
Contact:    dominik.greuter@proton.me
"""

import numpy as np
import os
import pandas as pd
import logging
import argparse 
import torch
import torch.nn as nn
import dgl
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, \
                            EarlyStopping
from Utils.graph_utils import GraphDataset, load_local_data, temps2graphs, \
                LocalMHN, predict_testset, decode_prediction, get_top_k_values,\
                temps2fps
from Utils.model_utils import LocalRetro
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s',
                              "%Y-%m-%d %H:%M:%S")
if not os.path.exists('log'):
    os.mkdir('log')
file_handler = logging.FileHandler('log/train_gnn.log', mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False


def parse_args():
    logger.debug(f'initializing args...')
    base_path = os.path.abspath('')
    parser = argparse.ArgumentParser(description="training LMHR",
                                    epilog="--")
    #default
    parser.add_argument("-i", "--localdir", 
                    default=os.path.join(base_path, 'data/USPTO_50K'),
                        type=str, help="""path to dir where local template files
                        (*_templates.csv and labeled_data.csv) are stored""")
    #  use this for developing
    #  parser.add_argument("-i", "--localdir", 
                    #  default=os.path.join(base_path, 'data/USPTO_50K/reduced_set'),
                        #  type=str, help="""path to dir where local template files
                        #  (*_templates.csv and labeled_data.csv) are stored""")
    parser.add_argument("-o", "--output_path", 
                    default=os.path.join(base_path, 'output'),  type=str, 
                       help="path for output files")
    parser.add_argument("-af", "--atom_featurizer", 
                        default=CanonicalAtomFeaturizer(),
                        help="dgllife Featurizer for graph featurization")
    parser.add_argument("-bf", "--bond_featurizer", 
                        default=CanonicalBondFeaturizer(),
                        help="dgllife Featurizer for graph featurization")
    parser.add_argument("-fg", '--force_graphs', default=False,
                       help='forces graph creation')
    parser.add_argument("-ft", '--force_training', default=True,
                       help='forces model training')
    parser.add_argument("-fp", '--force_prediction', default=True,
                       help='''forces testset prediction,
                        overwrites raw_prediction.txt and 
                        decode_prediction.txt''')
    parser.add_argument("-et", '--encoder_type', default='linear_fps',
                        help='type of encoder in mhn, options: '
                            '"MPNN", "MPNN_wo_GRU", "Baseline" '
                            '"linear_fps"')
    parser.add_argument("-ge", '--graph_encoder', default='concat',
                        help='type of temp encoder in mhn, options: '
                            '"reactant", "product", "concat"')
    parser.add_argument("--hidden_size", default=32)
    parser.add_argument("--out_size", default=32)#128
    parser.add_argument("--device", default=None)
    parser.add_argument("--beta", default=0.05)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--n_passing", default=6)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--n_epochs", default=1)
    parser.add_argument("--patience", default=3)
    parser.add_argument("--top_k", default = 50)
    parser.add_argument("--GPU", default = 2)
    parser.add_argument("--fp_size", default = 2048)
    parser.add_argument("--fp_radius", default = 2)
    parser.add_argument("--fp_type", default = 'rdk+morgan')
    parser.add_argument("--fp_jobs", default = 1)
    args = parser.parse_args()
    logger.debug(f'initializing args done')
    logger.debug(f'args: {vars(args)}\n')
    return args

def check_paths(args):
    if not os.path.exists(args.localdir):
        raise Exception(f"path to template {args.localdir} does not exist")
    if not os.path.exists(args.output_path):
        logger.info(f'output path {args.output_path} did not exist, but does now!')
        os.mkdir(args.output_path)

def set_device(args):
    args.device = f'cuda:{args.GPU}' if torch.cuda.is_available() else 'cpu'
    logger.debug(f'computing on device: {args.device}')


#copied and adapted from LocalRetro
def collate_graphdata(batch):
    '''returns seperate atom and bond labels with len: feat_size x len(bg)
        all entries are 0 but temp_id at edit_site'''
    graphs,  labels, _ = zip(*batch)
    atom_labels = []
    bond_labels = []
    for graph, label in zip(graphs, labels):
        atom_label = torch.zeros(graph.number_of_nodes())
        bond_label = torch.zeros(graph.number_of_edges())
        for l in label:
            if l[0] is 'a':
                atom_label[l[1]] = l[2]
            if l[0] is 'b':
                bond_label[l[1]] = l[2] 
        atom_labels.append(atom_label)
        bond_labels.append(bond_label)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg, torch.cat(atom_labels), torch.cat(bond_labels)

#copied and adapted from LocalRetro
def collate_testdata(batch):
    graphs,  _ , prod_smiles = zip(*batch)
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg, prod_smiles

if __name__=="__main__":
    logger.debug(3*'\n')

    args = parse_args()
    check_paths(args)
    if args.encoder_type not in ['MPNN', 'MPNN_wo_GRU', 'Baseline', 
                                 'linear_fps']:
        raise ValueError('invalid encoder type, please check valid options')
    args.raw_path = os.path.join(args.output_path, 'raw_predictions.txt')
    set_device(args)
    torch.set_default_dtype(torch.float64)


    X, y, atom_templates, bond_templates  = load_local_data(args) 
    args.zero_temp = {0:'CC>>CC'}
    templates = {**args.zero_temp, **atom_templates, **bond_templates}
    args.n_temps = len(templates)
    for split in ['train', 'val', 'test']:
        logger.debug(f"X {split} len {X[X['Split']==split].count()}")

    if args.encoder_type in ['MPNN', 'MPNN_wo_GRU', 'Baseline']:
        templates = temps2graphs(args, templates)

    elif args.encoder_type == 'linear_fps':
        templates = temps2fps(args, templates)

    train_dataset = GraphDataset(args, X, y,  'train')
    val_dataset   = GraphDataset(args, X, y,  'val')
    test_dataset  = GraphDataset(args, X, y,  'test')
    collate_fn = collate_graphdata

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                                shuffle=True, collate_fn=collate_testdata)

    if args.encoder_type == 'Baseline':
        model = LocalRetro(args)
    else:
        model= LocalMHN(args, templates)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    stopper = EarlyStopping(mode='lower', patience=args.patience,
                           filename='best_model.pth')

    if not os.path.exists('best_model.pth') or args.force_training:
        logger.info('starting training')
        for epoch in range(args.n_epochs):
            epoch_loss = 0
            tic = time()
            for i, (bg, atom_labels, bond_labels) in enumerate(train_loader):
                optimizer.zero_grad()
                atom_labels = atom_labels.to(args.device).to(torch.long)
                bond_labels = bond_labels.to(args.device).to(torch.long)
                
                #predict
                bg = bg.to(args.device)
                node_feats = bg.ndata.pop('h').to(args.device).to(torch.float64)
                edge_feats = bg.edata.pop('e').to(args.device).to(torch.float64)
                out_atom, out_bond = model(bg, node_feats, edge_feats)

                loss_atom = criterion(out_atom, atom_labels)
                loss_bond = criterion(out_bond, bond_labels)
                total_loss = loss_atom + loss_bond
                epoch_loss += total_loss.item()
                total_loss.backward()
                optimizer.step()
            toc = time()
            logger.info(f'epoch \t{epoch+1}/{args.n_epochs} avg train loss: '
                    f'\t{(epoch_loss/len(train_loader)):.4f} in {toc-tic:.0f}s')

            with torch.no_grad():
                val_loss = 0
                for bg, atom_labels, bond_labels in val_loader:
                    atom_labels = atom_labels.to(args.device).to(torch.long)
                    bond_labels = bond_labels.to(args.device).to(torch.long)
                    
                    #predict
                    bg = bg.to(args.device)
                    node_feats = bg.ndata.pop('h').to(args.device).to(torch.float64)
                    edge_feats = bg.edata.pop('e').to(args.device).to(torch.float64)
                    out_atom, out_bond = model(bg, node_feats, edge_feats)

                    loss_atom = criterion(out_atom, atom_labels)
                    loss_bond = criterion(out_bond, bond_labels)
                    total_loss = loss_atom + loss_bond
                    val_loss += total_loss.item()
                stop = stopper.step(val_loss, model)
                logger.info(f'epoch \t{epoch+1}/{args.n_epochs} \tval loss: '
                            f'\t{(val_loss/len(val_loader)):.4f}')
            if stop:
                logger.info(f'early stopped')
                break

    if not os.path.exists(args.raw_path) or args.force_prediction:
        logger.info(f'loading best model and starting prediction')
        if os.path.exists('best_model.pth'):
            if args.encoder_type in 'Baseline':
                model = LocalRetro(args)
            else:
                model = LocalMHN(args, templates)
            stopper.load_checkpoint(model)

        else:
            logger.warning(f'did not find a trained model')
        predict_testset(args, model, test_loader)
    if not os.path.exists(os.path.join(args.output_path, 
                'decoded_prediction.txt')) or args.force_prediction:
        logger.info(f'decoding prediction')
        decode_prediction(args)
    logger.info(f'calculating top k')
    get_top_k_values(args)
