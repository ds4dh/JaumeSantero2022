# Load argument packages
import argparse
import re

# Load transformer package
from onmt.translate.translator import Translator
from onmt.translate import GNMTGlobalScorer
from onmt.model_builder import load_test_model
import onmt.opts as opts

# Load data science packages
import numpy as np
import torch

# Load chemical packages
from rdkit.Chem import Descriptors, Descriptors3D, MolFromSmiles, Lipinski

# Path to model
MODEL = '../available_models/MIT_mixed_augm/MIT_mixed_augm_model_average_20.pt'

# Set number of predicted products
number_of_products = 3

# From SMILES to tokens
def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

# Get molecule descriptors
def get_descriptors_from_mol(mol_obj, descriptors_list, random_seed=0):

    descriptors_dict = {k: None for k in descriptors_list}
    for k in descriptors_list:
        try:
            if hasattr(Descriptors, k):
                descriptors_dict[k] = getattr(Descriptors, k)(mol_obj)
                continue

            if hasattr(Descriptors3D, k):
                hmol_obj = AllChem.AddHs(mol_obj)
                AllChem.EmbedMolecule(hmol_obj, useExpTorsionAnglePrefs=True,
                useBasicKnowledge=True, randomSeed=random_seed)
                AllChem.UFFOptimizeMolecule(hmol_obj)
                descriptors_dict[k] = getattr(Descriptors3D, k)(hmol_obj)
                continue

            if hasattr(Lipinski, k):
                descriptors_dict[k] = getattr(Lipinski, k)(mol_obj)

            else:
                raise NotImplementedError

        except:
                descriptors_dict[k] = None

    return descriptors_dict

# Reaction prediction function
def reactionPrediction(translator, reac_smi):
    
    """    
        Input:
            Model translator:
                translator (object)
            Reactants and reagents in SMILES
                reac_smi (str)                
                Example: reac_smi = 'N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F>C1CCOC1.[H-].[Na+]'
                
        Return:
            Scores and products in SMILES:
                (list (float32), (list (str))
                Example: ([tensor(1.0000)], ['N#Cc1ccsc1Nc1cc(F)c(F)cc1[N+](=O)[O-]'])
            
        Footnote from Schwaller 2019:
            The product of the probabilities of all predicted
            tokens are used as a confidence score
    """

    # Tokenize SMILE molecules
    reac_tok = smi_tokenizer(reac_smi)

    # Output tokenized product
    scores, product_tok = translator.translate(src_data_iter=[reac_tok], batch_size=64)

    # Obtain SMILES product from tokenized product
    product_smi = [pred.replace(' ','') for pred in product_tok[0]]
    
    # Transform log-probs into probs
    scores = [torch.exp(score) for score in scores[0]]
        
    return scores, product_smi

# Display products and scores in terminal
def show_products(scores, products):
    print("-------------------------\n")
    print("Score\t\tProduct\n")
    print("-------------------------\n")
    for iproduct, product in enumerate(products):
        properties = get_descriptors_from_mol(MolFromSmiles(product), descriptors_list, random_seed=0)
        print("%.2e\t%s\n"%(scores[iproduct], product))
        print(properties)
        print("-------------------------\n")

# Loads model translator
def load_model(number_of_products=1):

    # Parsing model parameters
    parser = argparse.ArgumentParser(description='translate.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #opts.add_md_help_argument(parser)
    opts.translate_opts(parser)
    opt = parser.parse_args(['-model=%s'%MODEL,
                             '-src=%s'%'CCC',
                             '-batch_size=%s'%'64',
                             '-replace_unk',
                             '-max_length=%s'%'200'])
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    # Load transformer model
    fields, model, model_opt = load_test_model(opt) #, dummy_opt.__dict__)

    # Set score parameters
    scorer = GNMTGlobalScorer(opt.alpha, opt.beta,
                              opt.coverage_penalty,
                              opt.length_penalty)

    # Create dictionary with model parameters
    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam", #"report_bleu", "window_size", "window_stride", "mask_from",
                        "data_type", "replace_unk", "gpu", "verbose", #"fast", "sample_rate", "window", "image_channel_size",
                        ]}

    # Create transfomer
    translator = Translator(model, fields, global_scorer=scorer,
                            report_score=True, out_file=None,
                            copy_attn=model_opt.copy_attn, logger=None,
                            #log_probs_out_file=None,
                            n_best=number_of_products, **kwargs)
    
    return translator

# Load model
translator = load_model(number_of_products)

# Define reactants and reagents in SMILES
# reac_smi = 'COC(=O)Cc1cn(C)c2cc(O)ccc12.Cc1nn(-c2ccc(C(F)(F)F)cc2)cc1C(C)CO>CCCCP(CCCC)CCCC.Cc1ccccc1'
# reac_smi = 'COC(=O)Cc1cn(C)c2cc(O)ccc12.Cc1nn(-c2ccc(C(F)(F)F)cc2)cc1C(C)CO.CCCCP(CCCC)CCCC.Cc1ccccc1'
# reac_smi = 'C1=COCC1.COC(=O)c1ccc(I)cc1>CC(=O)[O-].CC(=O)[O-].CC(=O)[O-].CCCCCC.CCCC[N+](CCCC)(CCCC)CCCC.CCOC(C)=O.CN(C)C=O.COC(C)(C)C.O.O.O.[Cl-].[Li+].[Pd+2].c1ccc(P(c2ccccc2)c2ccccc2)cc1'
# reac_smi = 'CCOC(=O)CC(=O)[O-].O=C(O)c1cc(I)ccc1F>C1CCC2=NCCCN2CC1.C1CCOC1.CC#N.C[Si](C)(C)Cl.O=C(c1ncc[nH]1)c1ncc[nH]1.[K+]'
# reac_smi = 'CCOC(=O)CC(=O)[O-].O=C(O)c1cc(I)ccc1F.C1CCC2=NCCCN2CC1.C1CCOC1.CC#N.C[Si](C)(C)Cl.O=C(c1ncc[nH]1)c1ncc[nH]1.[K+]'
# reac_smi = 'N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F>C1CCOC1.[H-].[Na+]'
reac_smi = 'N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F.C1CCOC1.[H-].[Na+]'
# reac_smi = 'Clc1ccc2ncccc2c1.OBO.Cc1ccc2c(cnn2C2CCCCO2)c1-c1ccc2ncccc2c1'

# Run model for a given reaction
scores, products = reactionPrediction(translator, reac_smi)

for i in range(len(products)):
    print('%.4f - %s\n'%(scores[i], products[i]))

# Get descriptors
#descriptors_list = ["MolLogP", "SlogP_VSA1", "Asphericity", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors"]

# Show results in terminal
#show_products(scores, products)
