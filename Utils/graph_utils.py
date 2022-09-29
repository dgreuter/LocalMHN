import os
from os.path import join
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit.Chem import MolFromSmarts, MolFromSmiles
from tqdm import tqdm
from dgllife.utils import smiles_to_bigraph, mol_to_bigraph
from dgl.data.utils import save_graphs, load_graphs
import dgl
import dgl.function as fn
from dgl.utils import expand_as_pair
import numpy as np
from .molutils import convert_smiles_to_fp
from .template_decoder import read_prediction, decode_localtemplate, \
                        get_MaxFrag, isomer_match, demap
from .model_utils import LocalRetro

import logging
logger = logging.getLogger()


def remove_template_annotation(templates_dict):
    for key in templates_dict:
        templates_dict[key] = templates_dict[key].split('_')[0]
    return templates_dict

def load_local_data(args, input_col='Products', split_col='Split'):
    '''
    loads data from labeled_data.csv 
    output:     X, y as df 
                templates as dict 
    '''
    logger.info('loading preprocessed local data')
    logger.debug(f'args: {vars(args)}')
    df = pd.read_csv(os.path.join(args.localdir, 'labeled_data.csv'))
    logger.debug(f'loaded labeled data with info, len {len(df)}')
    df = df[df.Mask > 0]
    logger.debug(f'removed entries with mask 0, new len {len(df)}')

    # loading data as df
    X = df.loc[:, [input_col, split_col]]
    y = df.loc[:, ['Labels']]

    df = pd.read_csv(os.path.join(args.localdir, 'atom_templates.csv'), 
                     index_col='Class')
    atom_templates = {idx:df.loc[idx]['Template'] for idx in df.index}
    atom_templates = remove_template_annotation(atom_templates)
   
    df = pd.read_csv(os.path.join(args.localdir, 'bond_templates.csv'), 
                     index_col='Class')
    bond_templates = {idx:df.loc[idx]['Template'] for idx in df.index}
    bond_templates = remove_template_annotation(bond_templates)
    logger.info('done loading preprocessed local data')
    args.delimiter = df.index[0]
    logger.debug(f'load data temp delimiter: {args.delimiter}')
    return X, y, atom_templates, bond_templates


def smiles2graphs(args, smiles, save_path, is_smarts=False): 
    graphs = []
    if is_smarts:
        converter = MolFromSmarts
    else:
        converter = MolFromSmiles
    for SM in tqdm(smiles):
        mol = converter(SM)
        mol.UpdatePropertyCache(strict=False)
        graphs.append(mol_to_bigraph(mol, #add_self_loop=True,
                             node_featurizer=args.atom_featurizer,
                             edge_featurizer=args.bond_featurizer))
    logger.info(f'created and saved graphs to {save_path}')
    save_graphs(save_path, graphs)
    return graphs


def temps2graphs(args, templates):
    '''
    input: templates as reaction smarts
    output: list of temps (atom, bond) seperated by args.delimiter
            each temp is tuple: (reactant_side, product_side)
    '''
    logger.info('creating template graphs')
    reaction_side = ['product', 'reactant']
    graphs = []
    for j, side in enumerate(reaction_side):
        temp_path = os.path.join(args.localdir, \
                                 f'temp_graphs_{side}.bin')
        if not os.path.exists(temp_path) or args.force_graphs:
            smarts = [templates[key].split('>>')[j] for key in templates]
            graphs.append(smiles2graphs(args, smarts, temp_path, is_smarts=True))
        else:
            logger.debug(f'graphs exist for {side} side, loading from {temp_path}')
            gs, ld = load_graphs(temp_path)
            graphs.append(gs)
    product_graphs  = graphs[0]
    reactant_graphs = graphs[1]
    logger.info(f"done loading template graphs")
    return product_graphs 


def temps2fps(args, templates):
    logger.info('creating template fingerprints')
    split_template_list = [templates[key].split('>')[0] for key in templates]
    templates_np = convert_smiles_to_fp(split_template_list, is_smarts=True,
                    fp_size=args.fp_size, radius=args.fp_radius, 
                    which=args.fp_type, njobs=args.fp_jobs)

    split_template_list = [templates[key].split('>')[-1] for key in templates]
    reactants_np = convert_smiles_to_fp(split_template_list, is_smarts=True, 
                    fp_size=args.fp_size, radius=args.fp_radius, 
                    which=args.fp_type, njobs=args.fp_jobs)

    templates = templates_np-(reactants_np*0.5)
    assert templates.shape[0] == len(np.unique(templates, axis=1))
    templates = torch.from_numpy(templates).to(args.device)
    logger.info(f"done creating template fingerprints")
    logger.info(f'template dim: {templates.shape}')
    return templates


class GraphDataset:
    '''
    input:  from load_local_data
            X, y    dfs
            split   string

    output: feature molecule graphs
            labels as tuple (temp_type, site, temp_number)
    '''
    def __init__(self, args, X, y, split):
        logger.info(f'creating {split} graph dataset')        
        self.ids = X.index[X['Split'] == split].values
        self.smiles = X[X['Split']==split]['Products'].values
        labels = y.iloc[self.ids]['Labels']
        self.labels= [eval(l) for l in labels]
        logger.info(f'loaded dset with len X:{len(self.smiles)} '
                        f'y:{len(self.labels)}')

        logger.info('loading/creating graphs for feature molcules')
        feature_path = os.path.join(args.localdir, f'{split}_graphs.bin')
        if not os.path.exists(feature_path) or args.force_graphs:
            self.graphs = smiles2graphs(args, self.smiles, feature_path)
        else:
            logger.warning(f'graphs exist, loading from {feature_path}')
            self.graphs, label_dict = load_graphs(feature_path)
        logger.info(f'done creating {split} graph dataset')

    def __getitem__(self, item):
        return self.graphs[item], self.labels[item], self.smiles[item]

    def __len__(self):
        return len(self.labels)
        

class LocalMHN(nn.Module):
    def __init__(self, args, templates):
        super().__init__()
        self.beta = args.beta
        if args.encoder_type in ['MPNN', 'MPNN_wo_GRU']:
            self.template_encoder = LocalRetro(args, pool=True).to(args.device)
            self.t_g = dgl.batch(templates).to(args.device)
            self.t_n = self.t_g.ndata.pop('h').to(args.device).to(torch.float64)
            self.t_e = self.t_g.edata.pop('e').to(args.device).to(torch.float64)
        elif args.encoder_type == 'linear_fps':
            self.t_g = None
            self.template_encoder = nn.Sequential(
            nn.Linear(args.fp_size, args.out_size),
            nn.ReLU(),
            nn.LayerNorm(args.out_size)).to(args.device)
            self.templates = templates
        self.molecule_encoder = LocalRetro(args).to(args.device)


    def forward(self, m, node_feats, edge_feats):
        if self.t_g:
            X_atom, X_bond = self.template_encoder(self.t_g, self.t_n, self.t_e) # n_temps x args.out_size
        else:
            X_atom = self.template_encoder(self.templates) # n_temps x args.out_size
            X_bond = X_atom
        Xi_atom, Xi_bond = self.molecule_encoder(m, node_feats, edge_feats) # len(batch_graph) x args.out_size
        XXi_atom = Xi_atom @ X_atom.T # len(batch_graph) x n_temps
        XXi_bond = Xi_bond @ X_bond.T
        return self.beta*XXi_atom, self.beta*XXi_bond


#copied and modified from localretro
def get_id_template(a, class_n):
    class_n = class_n # no template
    edit_idx = a//class_n
    template = a%class_n
    return (edit_idx, template)

#copied and modified from localretro
def output2edit(out, top_num):
    class_n = out.size(-1) #n_temps
    readout = out.cpu().detach().numpy()
    readout = readout.reshape(-1)
    output_rank = np.flip(np.argsort(readout))
    output_rank = [r for r in output_rank 
                   if get_id_template(r, class_n)[1] != 0][:top_num]
    selected_edit = [get_id_template(a, class_n) 
                                        for a in output_rank]
    selected_proba = [readout[a] for a in output_rank]
    return selected_edit, selected_proba

#copied and modified from localretro
def combined_edit(atom_out, bond_out, top_num):
    edit_id_a, edit_proba_a = output2edit(atom_out, top_num)
    edit_id_b, edit_proba_b = output2edit(bond_out, top_num)
    edit_id_c = edit_id_a + edit_id_b
    edit_type_c = ['a'] * top_num + ['b'] * top_num
    edit_proba_c = edit_proba_a + edit_proba_b
    edit_rank_c = np.flip(np.argsort(edit_proba_c))[:top_num]
    edit_type_c = [edit_type_c[r] for r in edit_rank_c]
    edit_id_c = [edit_id_c[r] for r in edit_rank_c]
    edit_proba_c = [edit_proba_c[r] for r in edit_rank_c]
    return edit_type_c, edit_id_c, edit_proba_c

#copied and modified from localretro
def predict_testset(args, model, test_loader):
    top_num = 50
    model.eval()
    with open(args.raw_path, 'w') as f:
        f.write('Test_id\tProducts\t%s\n' % '\t'.join(['Edit %s\tProba %s' % (i+1, i+1) \
                                                       for i in range(top_num)]))
        logger.info(f'Writing test molecules...') 
        with torch.no_grad():
            for batch_id, (bg,  prod_smiles) in enumerate(tqdm(test_loader)):
                #predict
                bg = bg.to(args.device)
                node_feats = bg.ndata.pop('h').to(args.device).to(torch.float64)
                edge_feats = bg.edata.pop('e').to(args.device).to(torch.float64)
                batch_atom_logits, batch_bond_logits = model(bg, node_feats, edge_feats)
                batch_atom_logits = F.softmax(batch_atom_logits,dim=1)
                batch_bond_logits = F.softmax(batch_bond_logits, dim=1) 
                sg = bg.remove_self_loop()
                graphs = dgl.unbatch(sg)
                nodes_sep = [0]
                edges_sep = [0]
                for g in graphs:
                    nodes_sep.append(nodes_sep[-1] + g.num_nodes())
                    edges_sep.append(edges_sep[-1] + g.num_edges())
                nodes_sep = nodes_sep[1:]
                edges_sep = edges_sep[1:]

                start_node = 0
                start_edge = 0
                for single_id, (graph, end_node, end_edge) \
                            in enumerate(zip(graphs, nodes_sep, edges_sep)):
                    _, edit_id, edit_proba = combined_edit( 
                                batch_atom_logits[start_node:end_node,:], 
                        batch_bond_logits[start_edge:end_edge,:], top_num)
                    start_node = end_node
                    start_edge = end_edge
                    f.write('%s\t%s\n' % (prod_smiles[single_id], \
                        '\t'.join(['(%s,%.3f)' % (edit_id[i], edit_proba[i]) \
                                   for i in range(top_num)])))
    logger.info(f'written results to {args.raw_path} with ')


#copied from localretro and modified
def decode_prediction(args):
    prediction_file = args.raw_path
    prediction = pd.read_csv(args.raw_path, sep = '\t')

    output_path = join(args.output_path, 'decoded_prediction.txt')

    atom_templates = pd.read_csv(join(args.localdir, 'atom_templates.csv'),
                                index_col='Class')
    bond_templates = pd.read_csv(join(args.localdir, 'bond_templates.csv'),
                                index_col='Class')
    template_infos = pd.read_csv(join(args.localdir, 'template_infos.csv'))
        
    atom_templates = {i: atom_templates['Template'][i] 
                            for i in atom_templates.index}
    bond_templates = {i: bond_templates['Template'][i] 
                            for i in bond_templates.index}
    template_infos = {template_infos['Template'][i]: {'edit_site': 
                            eval(template_infos['edit_site'][i]),
                        'change_H': eval(template_infos['change_H'][i]),
                        'change_C': eval(template_infos['change_C'][i]),
                        'change_S': eval(template_infos['change_S'][i])} 
                      for i in template_infos.index}

    raw_predictions = {}
    with open(prediction_file, 'r') as f:
        for line in f.readlines():
            seps = line.split('\t')
            if seps[0] == 'Test_id':
                continue
            raw_predictions[seps[0]] = seps[1:]

    result_dict = {}
    for test_id in tqdm(raw_predictions):
        predictions = raw_predictions[test_id]
        all_prediction = []
        for prediction in predictions:
            mol, pred_site, template, template_info, score =\
                    read_prediction(args, test_id, prediction,
                            atom_templates, bond_templates, template_infos)
            local_template = '>>'.join(['(%s)' % smarts for smarts in 
                                        template.split('_')[0].split('>>')])
            decoded_smiles = decode_localtemplate(mol, pred_site, 
                                            local_template, template_info)
            if decoded_smiles == None or str((decoded_smiles, score)) \
                                        in all_prediction:
                    continue
            all_prediction.append(str((decoded_smiles, score)))

            if len (all_prediction) >= args.top_k:
                break
        result_dict[test_id] = all_prediction
    with open(output_path, 'w') as f:
        for prod in result_dict :
            f.write('\t'.join([str(prod)] + result_dict[prod]) + '\n')
    logger.info(f'written decoded predictions')
    

#copied from localretro and modified
def get_top_k_values(args):
    df = pd.read_csv(join(args.localdir, 'labeled_data.csv'))
    df = df[df.Mask > 0]
    df = df.loc[:, ['Reactants', 'Products', 'Split']]
    df = df[df['Split']=='test']
    reactants = {df['Products'][i] : df['Reactants'][i] for i in df.index}

    result_file = join(args.output_path, 'decoded_prediction.txt')

    ground_truth = {prod: demap(MolFromSmiles(reactants[prod])) 
                    for prod in reactants}
    ground_truth_MaxFrag = {g: get_MaxFrag(ground_truth[g]) for g 
                            in ground_truth}

    results = {}       
    results_MaxFrag = {}
    with open(result_file, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.split('\n')[0]
            product = line.split('\t')[0]
            predictions = line.split('\t')[1:]
            MaxFrags = []
            results[product] = [eval(p)[0] for p in predictions]
            for p in results[product]:
                if p not in MaxFrags:
                    MaxFrags.append(get_MaxFrag(p))
            results_MaxFrag[product] = MaxFrags

    Exact_matches = []
    MaxFrag_matches = [] # Only compares the largest reactant fragment

    Exact_matches_multi = []
    MaxFrag_matches_multi = [] 
    for i in tqdm(results):
        match_exact = isomer_match(results[i], ground_truth[i])
        match_maxfrag = isomer_match(results_MaxFrag[i], ground_truth_MaxFrag[i])
        if len(i.split('.')) > 1:
            Exact_matches_multi.append(match_exact)
            MaxFrag_matches_multi.append(match_maxfrag)
        Exact_matches.append(match_exact)
        MaxFrag_matches.append(match_maxfrag)

    ks = [1, 3, 5, 10, 50]
    exact_k = {k:0 for k in ks}
    MaxFrag_k = {k:0 for k in ks}

    for i in range(len(Exact_matches)):
        for k in ks:
            if Exact_matches[i] <= k and Exact_matches[i] != -1:
                exact_k[k] += 1
            if MaxFrag_matches[i] <= k and MaxFrag_matches[i] != -1:
                MaxFrag_k[k] += 1

    for k in ks:
        logger.info ('Top-%d Exact accuracy: %.9f, MaxFrag accuracy: %.9f' % (k, exact_k[k]/len(Exact_matches), MaxFrag_k[k]/len(MaxFrag_matches)))
