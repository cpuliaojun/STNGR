"""
Take an input SMILES file, and augment it by some fixed factor via SMILES
enumeration.
"""

import argparse
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

# set working directory
# git_dir = os.path.expanduser("~/git/NPS-generation")
# python_dir = git_dir + "/python"
# os.chdir(python_dir)

# python_dir = './'
# os.chdir(python_dir)


from SmilesEnumerator import SmilesEnumerator
from utils import read_smiles, write_smiles
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
# def smi_to_scaffold(smi):
#     mol = Chem.MolFromSmiles(smi)
#     scaffold = GetScaffoldForMol(mol)
#     if scaffold:
#         return Chem.MolToSmiles(scaffold)
#     else:
#         return smi

def get_mol_chain_aroma(smi):
    mol = Chem.MolFromSmiles(smi)
    chain = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol),
                             isomericSmiles=False,
                             kekuleSmiles=False,
                             doRandom=False)
    return chain

def aug_data(aug_factor):
    ### CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,default='./../data/cleaned_HighResNpsTrain.txt')
    parser.add_argument('--output_file', type=str,default=f'./../data/nps/nps_train_smiles_generic_scaffold_{aug_factor}.txt')
    parser.add_argument('--enum_factor', type=int,
                        help='factor to augment the dataset by',default=aug_factor)
    args = parser.parse_args()
    print(f'procesing {aug_factor} data')




    # check output directory exists
    output_dir = os.path.dirname(args.output_file)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # read SMILES
    smiles = read_smiles(args.input_file)
    # convert to numpy array
    smiles = np.asarray(smiles)

    # create enumerator
    sme = SmilesEnumerator(canonical=False, enum=True)

    # also store and write information about enumerated library size
    summary = pd.DataFrame()

    # enumerate potential SMILES
    enum = []
    max_tries = 200 ## randomized SMILES to generate for each input structure
    for sm_idx, sm in enumerate(tqdm(smiles)):
        tries = []
        for try_idx in range(max_tries):
            this_try = sme.randomize_smiles(sm)
            tries.append(this_try)
            tries = [rnd for rnd in np.unique(tries)]
            if len(tries) > args.enum_factor:
                tries = tries[:args.enum_factor]
                break
        enum.extend(tries)
    # dic = {smi:Chem.MolToSmiles(GetScaffoldForMol(Chem.MolFromSmiles(smi)))for smi in enum }
    smi_list, scaffold_list = [],[]
    for smi in enum:
        scaffold = get_mol_chain_aroma(smi)
        if scaffold:
            smi_list.append(smi)
            scaffold_list.append(scaffold)
    df = pd.DataFrame({'smiles':smi_list,'genric_scaffold':scaffold_list})
    df.to_csv(args.output_file,header=None,index=False)


    # write to line-delimited file
    # write_smiles(enum, args.output_file)
    # print("wrote " + str(len(enum)) + " SMILES to output file: " + \
    #       args.output_file)



if __name__ == "__main__":
    # augment = [ 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
    # for aug_factor in augment:
    #     aug_data(aug_factor)
    # data = read_smiles('./../data/cleaned_HighResNpsTrain.txt')
    # smi_list, scaffold_list = [],[]
    # for smi in data:
    #     scaffold = get_mol_chain_aroma(smi)
    #     if scaffold:
    #         smi_list.append(smi)
    #         scaffold_list.append(scaffold)
    # print(len(scaffold_list))
    # scaffold_list = np.unique(scaffold_list).tolist()
    # print(len(scaffold_list))
    # df = pd.DataFrame({'smiles':smi_list,'scaffold':scaffold_list})
    # df.to_csv('./../data/augment_data/smiles_scaffold_1.txt',header=None,index=False)
    # with open('./../data/Train_unique.txt','w') as f:
    #     for smi in smi_list:
    #         f.write(smi+'\n')
    aug_data(100)


