"""
this code and the imported packages from pre_utils are copied from:
LocalRetro 
https://github.com/kaist-amsg/LocalRetro 
and were modified by:
Dominik Greuter
Contact:    dominik.greuter@proton.me
"""
import numpy as np
from collections import defaultdict
import pandas as pd
import errno, sys, os, re, copy, shutil
from argparse import ArgumentParser
import logging
import hashlib
from tqdm import tqdm
import rdkit
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdChemReactions
RDLogger.DisableLog('rdApp.*')

sys.path.append('../')
from pre_utils import extract_from_reaction


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
if not os.path.exists('log/'):
    os.mkdir('log')
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s - %(message)s',
                              "%Y-%m-%d %H:%M:%S")
file_handler = logging.FileHandler('log/preprocessing.log', mode='w')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False

            
def build_template_extractor(args):
    logger.debug(f'entering build_template_extractor' )
    setting = {'verbose': False, 'use_stereo': False, 'use_symbol': False, 'max_unmap': 5, 'retro': False, 'remote': True, 'least_atom_num': 2}
    for k in setting.keys():
        if k in args.keys():
            setting[k] = args[k]
    if args['retro']:
        setting['use_symbol'] = True
    logger.info(f'Template extractor setting: {setting}')
    logger.debug('calling extract_from_reaction')
    return lambda x: extract_from_reaction(x, setting)

def get_reaction_template(extractor, rxn, _id = 0):
    rxn = {'reactants': rxn.split('>>')[0], 'products': rxn.split('>>')[1], '_id': _id}
    result = extractor(rxn)
    return rxn, result

def get_full_template(template, H_change, Charge_change, Chiral_change):
    H_code = ''.join([str(H_change[k+1]) for k in range(len(H_change))])
    Charge_code = ''.join([str(Charge_change[k+1]) for k in range(len(Charge_change))])
    Chiral_code = ''.join([str(Chiral_change[k+1]) for k in range(len(Chiral_change))])
    if Chiral_code == '':
        return '_'.join([template, H_code, Charge_code])
    else:
        return '_'.join([template, H_code, Charge_code, Chiral_code])

def split_data_df(data, val_frac=0.1, test_frac=0.1, shuffle=False, seed=None):
    """edited from https://github.com/connorcoley/retrosim/blob/master/retrosim/data/get_data.py"""
    # Define shuffling
    logger.debug(f'split data seed: {seed}')
    if shuffle:
        if seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(seed)
        def shuffle_func(x):
            np.random.shuffle(x)
    else:
        def shuffle_func(x):
            pass

    # Go through each class
    logger.info('analyzing reaction classes, found:')
    classes = sorted(np.unique(data['class']))
    for class_ in classes:
        indeces = data.loc[data['class'] == class_].index
        N = len(indeces)
        logger.info('{}\t reactions with class {}'.format(N, class_))

        shuffle_func(indeces)
        train_end = int((1.0 - val_frac - test_frac) * N)
        val_end = int((1.0 - test_frac) * N)

        for i in indeces[:train_end]:
            data.at[i, 'dataset'] =  'train'
        for i in indeces[train_end:val_end]:
            data.at[i, 'dataset'] =  'val'
        for i in indeces[val_end:]:
            data.at[i, 'dataset'] =  'test'
    logger.info('dataset was split into:')
    for item in data['dataset'].value_counts().to_string().split('\n'):    
        logger.info(item)

def get_retrosim_data():
    df = pd.read_csv('https://github.com/connorcoley/retrosim/raw/master/retrosim/data/data_processed.csv', index_col=0)
    split_data_df(df, seed=123)
    df.rename(columns={'rxn_smiles':'reaction_smiles'}, inplace=True)
    df.rename(columns={'dataset':'Split'}, inplace=True)
    return df    

def reduce_dataset(data, n_samples=200):
    n_train = round(n_samples*0.8)
    n_valid = round(n_samples*0.1)
    assert n_train + 2*n_valid == n_samples
    df = pd.DataFrame()
    df = data[data['Split']=='train'][:n_train]
    df = df.append(data[data['Split']=='val'][:n_valid])
    df = df.append(data[data['Split']=='test'][:n_valid])
    return df

def extract_templates(args, extractor, df):
    logger.debug('entering extract_templates')
    
    TemplateEdits = {}
    TemplateCs = {}
    TemplateHs = {}
    TemplateSs = {}
    TemplateFreq = defaultdict(int)
    templates_A = defaultdict(int)
    templates_B = defaultdict(int)
    
    reactions = df['reaction_smiles'].tolist()
    for i, reaction in enumerate(tqdm(reactions)):
        try:
            rxn, result = get_reaction_template(extractor, reaction, i)
            if 'reactants' not in result or 'reaction_smarts' not in result.keys():
                logger.warning('template extraction problem: id: {i}')
                continue
            reactant = result['reactants']
            template = result['reaction_smarts']
            edits = result['edits']
            H_change = result['H_change']
            Charge_change = result['Charge_change']
            if args['use_stereo']:
                Chiral_change = result['Chiral_change']
            else:
                Chiral_change = {}
            template_H = get_full_template(template, H_change, Charge_change, Chiral_change)
            if template_H not in TemplateHs.keys():
                TemplateEdits[template_H] = {edit_type: edits[edit_type][2] \
                                             for edit_type in edits}
                TemplateHs[template_H] = H_change
                TemplateCs[template_H] = Charge_change
                TemplateSs[template_H] = Chiral_change

            TemplateFreq[template_H] += 1

            if args['retro']:
                for edit_type, bonds in edits.items():
                    bonds = bonds[0]
                    if len(bonds) > 0:
                        if edit_type in ['A', 'R']:
                            templates_A[template_H] += 1
                        else:
                            templates_B[template_H] += 1

            else:
                for edit_type, bonds in edits.items():
                    bonds = bonds[0]
                    if len(bonds) > 0:
                        if edit_type != 'A':
                            templates_A['%s_%s' % (template_H, edit_type)] += 1
                        else:
                            templates_B['%s_%s' % (template_H, edit_type)] += 1

                
        except KeyboardInterrupt:
            logger.error('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            logger.exception(i, e)
            
    logger.info(f'total # of template: {len(TemplateFreq)}, ' 
                f'# of atom template: {len(templates_A)}, ' 
                f'# of bond template: {len(templates_B)}')

    if args['retro']:
        derived_templates = {'atom':templates_A, 'bond': templates_B}
    else:
        derived_templates = {'real':templates_A, 'virtual': templates_B}
        
    TemplateInfos = pd.DataFrame({'Template': k, 'edit_site':TemplateEdits[k], 'change_H': TemplateHs[k], 'change_C': TemplateCs[k], 'change_S': TemplateSs[k], 'Frequency': TemplateFreq[k]} for k in TemplateHs.keys())
    TemplateInfos.to_csv('%s/template_infos.csv' % args['output_dir'])
    logger.debug('exiting extract_templates')
    return derived_templates

def export_template(derived_templates, args):
    logger.info('exporting teamplates')
    c = 1
    for k in derived_templates.keys():
        local_templates = derived_templates[k]
        templates = []
        template_class = []
        template_freq = []
        sorted_tuples = sorted(local_templates.items(), key=lambda item: item[1])
        for t in sorted_tuples:
            templates.append(t[0])
            template_freq.append(t[1])
            template_class.append(c)
            c += 1
        template_dict = {templates[i]:i+1  for i in range(len(templates)) }
        template_df = pd.DataFrame({'Template' : templates, 
                        'Frequency' : template_freq, 'Class': template_class})
        template_df.to_csv('%s/%s_templates.csv' % (args['output_dir'], k))
    logger.info(f"templates written to {args['output_dir']}")
    return

def get_edit_site_retro(smiles):
    mol = Chem.MolFromSmiles(smiles)
    A = [a for a in range(mol.GetNumAtoms())]
    B = []
    for atom in mol.GetAtoms():
        others = []
        bonds = atom.GetBonds()
        for bond in bonds:
            atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            other = [a for a in atoms if a != atom.GetIdx()][0]
            others.append(other)
        b = [(atom.GetIdx(), other) for other in sorted(others)]
        B += b
    return A, B
    
def get_edit_site_forward(smiles):
    mol = Chem.MolFromSmiles(smiles)
    A = [a for a in range(mol.GetNumAtoms())]
    B = []
    for atom in mol.GetAtoms():
        others = []
        bonds = atom.GetBonds()
        for bond in bonds:
            atoms = [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
            other = [a for a in atoms if a != atom.GetIdx()][0]
            others.append(other)
        b = [(atom.GetIdx(), other) for other in sorted(others)]
        B += b
    V = []
    for a in A:
        V += [(a,b) for b in A if a != b and (a,b) not in B]
    return V, B

def labeling_dataset(args, split, template_dicts, template_infos, extractor, df):
    logger.debug('entering labeling_dataset')
    if os.path.exists('%s/preprocessed_%s.csv' % (args['output_dir'], split)) and args['force_templates'] == False:
        logger.info('%s data already preprocessed...loaded data!' % split)
        return pd.read_csv('%s/preprocessed_%s.csv' % (args['output_dir'], split))
    
    reactions = df[df['Split']==split]['reaction_smiles'].tolist()
    reactants = []
    products = []
    reagents = []
    labels = []
    frequency = []
    success = 0
    logger.debug(f'''extacted from df: num of reactions in 
                 {split} set: {len(reactions)}''')
    
    for i, reaction in enumerate(reactions):
        product = reaction.split('>>')[1]
        reagent = ''
        rxn_labels = []
        try:
            rxn, result = get_reaction_template(extractor, reaction, i)
            template = result['reaction_smarts']
            reactant = result['reactants']
            product = result['products']
            reagent = '.'.join(result['necessary_reagent'])
            edits = {edit_type: edit_bond[0] for edit_type, edit_bond in result['edits'].items()}
            H_change, Charge_change, Chiral_change = result['H_change'], result['Charge_change'], result['Chiral_change']
            if args['use_stereo']:
                Chiral_change = result['Chiral_change']
            else:
                Chiral_change = {}
            template_H = get_full_template(template, H_change, Charge_change, Chiral_change)
            
            if template_H not in template_infos.keys():
                reactants.append(reactant)
                products.append(product)
                reagents.append(reagent)
                labels.append(rxn_labels)
                frequency.append(0)
                continue
                
        except KeyboardInterrupt:
            logger.error('Interrupted')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
        except Exception as e:
            logger.exception(i, e)
            reactants.append(reactant)
            products.append(product)
            reagents.append(reagent)
            labels.append(rxn_labels)
            frequency.append(0)
            continue
        
        edit_n = 0
        for edit_type in edits:
            if edit_type == 'C':
                edit_n += len(edits[edit_type])/2
            else:
                edit_n += len(edits[edit_type])
            
            
        if edit_n <= args['max_edit_n']:
            try:
                success += 1
                if args['retro']:
                    atom_sites, bond_sites = get_edit_site_retro(product)
                    for edit_type, edit in edits.items():
                        for e in edit:
                            if edit_type in ['A', 'R']:
                                rxn_labels.append(('a', atom_sites.index(e), template_dicts['atom'][template_H]))
                            else:
                                rxn_labels.append(('b', bond_sites.index(e), template_dicts['bond'][template_H]))
                    reactants.append(reactant)
                    products.append(product)
                    reagents.append(reagent)       
                    labels.append(rxn_labels)
                    frequency.append(template_infos[template_H]['frequency'])
                else:
                    if len(reagent) != 0:
                        reactant = '%s.%s' % (reactant, reagent)
                    virtual_sites, real_sites = get_edit_site_forward(reactant)
                    for edit_type, bonds in edits.items():
                        for bond in bonds:
                            if edit_type != 'A':
                                rxn_labels.append(('r', real_sites.index(bond), template_dicts['real']['%s_%s' % (template_H, edit_type)]))
                            else:
                                rxn_labels.append(('v', virtual_sites.index(bond), template_dicts['virtual']['%s_%s' % (template_H, edit_type)]))
                    reactants.append(reactant)
                    products.append(reactant)
                    reagents.append(reagent)
                    labels.append(rxn_labels)
                    frequency.append(template_infos[template_H]['frequency'])
                
            except Exception as e:
                logger.exception(i,e)
                reactants.append(reactant)
                products.append(product)
                reagents.append(reagent)
                labels.append(rxn_labels)
                frequency.append(0)
                continue
                
            if i % 5000 == 0:
                logger.info('Processing  %s data..., success %s data (%s/%s)' %
                       (split, success, i, len(reactions)))
        else:
            logger.warning('Reaction # %s has too many edits (%s)...may be wrong mapping!' % (i, edit_n))
            reactants.append(reactant)
            products.append(product)
            reagents.append(reagent)
            labels.append(rxn_labels)
            frequency.append(0)
            
    logger.info('Derived tempaltes cover %.3f of %s data reactions' %
           ((success/len(reactions)), split))
    
    df = pd.DataFrame({'Reactants': reactants, 'Products': products, 'Reagents': reagents, 'Labels': labels, 'Frequency': frequency})
    df.to_csv('%s/preprocessed_%s.csv' % (args['output_dir'], split))
    logger.info(f"written preprcessed data to {args['output_dir']}")
    logger.debug('exiting labeling_dataset')
    return df

def make_simulate_output(args, split = 'test'):
    df = pd.read_csv('%s/preprocessed_%s.csv' % (args['output_dir'], split))
    with open('%s/simulate_output.txt' % args['output_dir'], 'w') as f:
        f.write('Test_id\tReactant\tProduct\t%s\n' % '\t'.join(['Edit %s\tProba %s' % (i+1, i+1) for i in range(args['max_edit_n'])]))
        for i in df.index:
            labels = []
            for y in eval(df['Labels'][i]):
                if y != 0:
                    labels.append(y)
            if len(labels) == 0:
                lables = [(0, 0)]
#             print (['%s\t%s' % (l, 1.0) for l in labels])
            string_labels = '\t'.join(['%s\t%s' % (l, 1.0) for l in labels])
            f.write('%s\t%s\t%s\t%s\n' % (i, df['Reactants'][i], df['Products'][i], string_labels))
    return 
    
    
def combine_preprocessed_data(train_pre, val_pre, test_pre, args):
    logger.debug('entering combine_preprocessed_data')
    train_valid = train_pre
    val_valid = val_pre
    test_valid = test_pre
    
    logger.info('combining labeled data')
    train_valid['Split'] = ['train'] * len(train_valid)
    val_valid['Split'] = ['val'] * len(val_valid)
    test_valid['Split'] = ['test'] * len(test_valid)
    all_valid = train_valid.append(val_valid, ignore_index=True)
    all_valid = all_valid.append(test_valid, ignore_index=True)
    all_valid['Mask'] = [int(f>=args['min_template_n']) for f in all_valid['Frequency']]
    logger.info(f'Valid data size: {len(all_valid)}')
    all_valid = all_valid[all_valid['Mask']!=0]
    logger.info(f'Valid data size after removing invalids: {len(all_valid)}')
    if args['reduced']:
        all_valid = reduce_dataset(all_valid)
    all_valid.to_csv('%s/labeled_data.csv' % args['output_dir'], index = None)
    logger.info(f"written combined data to {args['output_dir']}")
    logger.debug('exiting combine_preprocessed_data')
    return

def load_templates(args):
    logger.info('loading templates from csv file')
    template_dicts = {}
    if args['retro']:
        keys = ['atom', 'bond']
    else:
        keys = ['real', 'virtual']
        
    for site in keys:
        template_df = pd.read_csv('%s/%s_templates.csv' % (args['output_dir'], site))
        template_dict = {template_df['Template'][i]:template_df['Class'][i] for i in template_df.index}
        logger.info('loaded {len(template_dict)}, {site} templates')
        template_dicts[site] = template_dict
                                          
    template_infos = pd.read_csv('%s/template_infos.csv' % args['output_dir'])
    template_infos = {t: {'edit_site': eval(e), 'frequency': f} for t, e, f in zip(template_infos['Template'], template_infos['edit_site'], template_infos['Frequency'])}
    logger.info('loaded total {len(template_infos)} templates')
    return template_dicts, template_infos


def change_out_path(args):
    out_path = args['output_dir']
    logger.info(f'using reducet dataset')
    out_path = os.path.join(out_path, 'reduced_set')
    args['output_dir'] = out_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        logger.info(f'Created directory {out_path}')
    return args

def copy_templates(args):
    red_path = args['output_dir']
    #  red_path  = os.path.join(base_path, 'reduced_set/')
    base_path = os.path.dirname(red_path)
    #  red_path = os.path.abspath(red_path)
    logger.debug(f'red path {red_path}')
    shutil.copy(os.path.join(base_path, 'bond_templates.csv'), red_path)
    shutil.copy(os.path.join(base_path, 'atom_templates.csv'), red_path)
    shutil.copy(os.path.join(base_path, 'template_infos.csv'), red_path)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-r', '--retro', default=True,  help='''Retrosyntheis 
                        or forward synthesis (True for retrosnythesis)''')
    parser.add_argument('-v', '--verbose', default=False,  
                        help='Verbose during template extraction')
    parser.add_argument('-stereo', '--use-stereo', default=True,  
                        help='Use stereo info in template extraction')
    parser.add_argument('-ft', '--force-templates', default=False,  
                        help='Force template extraction')
    parser.add_argument('-fp', '--force-pre', default=True,  
                        help='Force to preprocess the dataset again')
    parser.add_argument('-m', '--max-edit-n', default=8,  
                        help='Maximum number of edit number')
    parser.add_argument('-min', '--min-template-n', type=int, default=1,  
                        help='Minimum of template frequency')
    parser.add_argument('--reduced', default=False, 
                        help='''uses first 200 samples and saves them to
                        [out_path]/reduced_set/''')
    args = parser.parse_args().__dict__
    args['output_dir'] = 'data/USPTO_50K'

    logger.debug(args)
    logger.debug(f"out dir{args['output_dir']}")
    logger.debug(f"stereo{args['use_stereo']}")
    return args
        

def main():
    args = parse_args()
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])

    if args['reduced']:
        args = change_out_path(args)
        if os.path.exists(os.path.join(os.path.dirname(args['output_dir']), 
                'bond_templates.csv')) and not os.path.exists(
                os.path.join(args['output_dir'],'reduced/bond_templates.csv')):
            copy_templates(args)

    if not os.path.exists(os.path.join(args['output_dir'], 'atom_templates.csv')) or\
    args['force_templates']:
        logger.info(f"templates not found in {args['output_dir']}, starting extraction")
        df = get_retrosim_data()
        df = df[df['Split']=='train']
        extractor = build_template_extractor(args)
        derived_templates = extract_templates(args, extractor, df)
        export_template(derived_templates, args)
        logger.info('finished template extraction')
    else:
        logger.info('templates exist, starting preprocessing')


    if not os.path.exists(os.path.join(args['output_dir'], 'labeled_data.csv'))\
       or args['force_pre'] :
        logger.info(''''no preprocessed data found in {args['output_dir']}, 
                    starting preprocessing''')
        df = get_retrosim_data()
        if args['reduced']:
            df = reduce_dataset(df, n_samples=200)
        template_dicts, template_infos = load_templates(args)
        extractor = build_template_extractor(args)
        test_pre = labeling_dataset(args, 'test', template_dicts,
                                    template_infos, extractor, df)
        make_simulate_output(args)
        val_pre = labeling_dataset(args, 'val', template_dicts, template_infos,
                                   extractor, df)
        train_pre = labeling_dataset(args, 'train', template_dicts, template_infos,
                                     extractor, df)
        combine_preprocessed_data(train_pre, val_pre, test_pre, args)
        logger.info('finished preprocessing data')
    else: 
        logger.info('''found preprocessed data in {out_path}
                    use -fp flag to reprocess''')


    
if __name__ == '__main__':  
    main()
