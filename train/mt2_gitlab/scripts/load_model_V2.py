# Load data science packages
import torch
import numpy as np
import pandas as pd

# Load argument packages
import argparse
import re

# Load transformer package
import onmt
import onmt.opts as opts
from onmt.translate import GNMTGlobalScorer
from onmt.model_builder import load_test_model
from onmt.translate.translator import Translator

# Load chemistry packages
import rdkit.Chem as Chem
from rdkit import RDLogger 
import rdkit.Chem.AllChem as AllChem
from rdkit.Chem import Descriptors, Descriptors3D, MolFromSmiles, Lipinski, AllChem     

# Non-verbose rdkit
RDLogger.DisableLog('rdApp.*')

# SMILES functions
canonicalize_smi = lambda smi: 'NA' if not Chem.MolFromSmiles(smi) else Chem.MolToSmiles(Chem.MolFromSmiles(smi))
equivalent_smi   = lambda smi: 'NA' if not Chem.MolFromSmiles(smi) else Chem.MolToSmiles(Chem.MolFromSmiles(smi), doRandom=True)

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

# Product prediction function
def productPrediction(translator, reac_smi):

    # Tokenize SMILE molecules
    reac_tok = smi_tokenizer(reac_smi)

    # Output tokenized product
    scores, product_tok = translator.translate_test(src=[reac_tok], batch_size=1)

    # Obtain SMILES product from tokenized product
    product_smi = [pred.replace(' ','') for pred in product_tok[0]]
    
    # Transform log-probs into probs
    scores = [torch.exp(score) for score in scores[0]]
        
    return scores, product_smi

# Reactants prediction function
def reactantsPrediction(translator, reac_smi):

    # Tokenize SMILE molecules
    reac_tok = smi_tokenizer(reac_smi)

    # Output tokenized product
    scores, reactants_tok = translator.translate_test(src=[reac_tok], batch_size=1)

    # Obtain SMILES product from tokenized product
    reactants_smi = [pred.replace(' ','') for pred in reactants_tok[0]]
    
    # Transform log-probs into probs
    scores = [torch.exp(score) for score in scores[0]]
        
    return scores, reactants_smi

# Loads model translator
def load_model(MODEL, number_of_solutions=1):

    # Parsing model parameters
    parser = argparse.ArgumentParser(description='translate.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
    fields, model, model_opt = load_test_model(opt)

    # Set score parameters
    scorer = GNMTGlobalScorer(opt.alpha, opt.beta,
                              opt.coverage_penalty,
                              opt.length_penalty)

    # Create dictionary with model parameters
    kwargs = {k: getattr(opt, k)
              for k in ["beam_size", "max_length", "min_length",
                        "stepwise_penalty", "block_ngram_repeat",
                        "ignore_when_blocking", "dump_beam",
                        "data_type", "replace_unk"]}

    # Create transfomer
    translator = Translator(model, fields=fields, global_scorer=scorer,
                            report_score=True, out_file=None,
                            copy_attn=model_opt.copy_attn, logger=None,
                            src_reader=onmt.inputters.str2reader["text"],
                            tgt_reader=onmt.inputters.str2reader["text"],
                            n_best=number_of_products, gpu=-1, **kwargs)
    
    return translator

# Get descriptors
descriptors_list = ["MolLogP", "SlogP_VSA1", "Asphericity", "TPSA", "MolWt", "NumHDonors", "NumHAcceptors"]

# Paths to models
MODEL_product   = '../available_models/MIT_mixed_augm/MIT_mixed_augm_model_average_20.pt'
MODEL_reactants = '../available_models/MIT_reactants_pred_x10/MIT_reactants_pred_x10_model_average_20.pt'

# Set number of predicted products
number_of_products  = 1
number_of_reactants = 1

# Generate translators for product and reactants prediction
translator_product   = load_model(MODEL_product, number_of_products)
translator_reactants = load_model(MODEL_reactants, number_of_reactants)

# Product prediction examples
smi_prod_pred = 'O=C(CCCN1CCC(NS(=O)(=O)c2ccccc2)CC1)c1ccccc1.[BH4-].[Na+]'
# smi_prod_pred = 'N#Cc1ccsc1N.O=[N+]([O-])c1cc(F)c(F)cc1F.C1CCOC1.[H-].[Na+]'
# smi_prod_pred = 'BrCC(C1)(C(C2=CC=CO2)O)CN1S(C3=CC=C(C)C=C3)(=O)=O.CO.O=C(O[K])O[K]'

# Reactants prediction examples
smi_reac_pred = 'N#Cc1ccsc1Nc1cc(F)c(F)cc1[N+](=O)[O-]'

# OPTIONAL: canonicalize molecules
smi_prod_pred_canon = '.'.join([canonicalize_smi(n) for n in smi_prod_pred.split('.')])
smi_reac_pred_canon = '.'.join([canonicalize_smi(n) for n in smi_reac_pred.split('.')])

# Run model for product prediction
scores_prod, products = productPrediction(translator_product, smi_prod_pred_canon)

# Run model for product prediction
scores_reac, reactants = reactantsPrediction(translator_reactants, smi_reac_pred_canon)

print(scores_prod, products)
print(scores_reac, reactants)

# Show product descriptors
print(get_descriptors_from_mol(MolFromSmiles(products[0]), descriptors_list, random_seed=0))

# Show descriptors for each reactant
for reactant in reactants[0].split('.'):
    print(get_descriptors_from_mol(MolFromSmiles(reactant), descriptors_list, random_seed=0))