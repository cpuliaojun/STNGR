from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
import pandas as pd
from utils import read_smiles
import os
import numpy as np


def get_mol_chain_aroma(smi):
    """

    Args:
        smi: 分子对应的SMILES

    Returns: generic scaffold

    """
    mol = Chem.MolFromSmiles(smi)

    chain = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol),
                             isomericSmiles=False,
                             kekuleSmiles=False,
                             doRandom=False)

    return chain

def get_bm_scaffold(smi):
    """

    Args:
        smi: 分子对应的SMILES

    Returns: bm scaffold

    """
    mol = Chem.MolFromSmiles(smi)
    bm = MurckoScaffold.GetScaffoldForMol(mol)
    chain = Chem.MolToSmiles(bm,
                             isomericSmiles=False,
                             kekuleSmiles=False,
                             doRandom=False)
    return chain

def get_mw(smi):
    mol = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(mol)
    return mw


moses = pd.read_csv('../data/moses_train.csv')
# moses.columns=['smiles']
moses['scaffold'] = moses['SMILES'].apply(lambda x:get_bm_scaffold(x))
# moses = moses.dropna(axis=0,how='any')
# moses = pd.DataFrame(moses['scaffold'].unique())
moses = moses[['SMILES','scaffold']]
moses.to_csv('./../data/moses_train_bm_scaffold_smiles.csv',header=None,index=False)


data = pd.read_csv('./../data/moses_train_bm_scaffold_smiles.csv',header=None)
data.columns =['smiles','scaffold']
data = data.dropna(axis=0,how='any')
data.to_csv('./../data/moses_train_bm_scaffold_smiles.csv',header=None,index=False)

data = pd.read_csv('./../data/test_scaffolds.csv')
data['mw'] = data['SMILES'].apply(lambda x:round(get_mw(x),1))
data = data.drop_duplicates(subset='mw')
data = data.sort_values(by='mw')
data.to_csv('./../data/moses_test_bm_scaffold_mw_unique.csv')

if __name__ == "__main__":
    # nps = pd.read_csv('./../data/cleaned_HighResNpsTrain.txt',header=None)
    # nps.columns = ['smiles']
    # nps['generic_scaffold'] = nps['smiles'].apply(lambda x:get_mol_chain_aroma(x))
    # os.mkdir('./../data/nps/')
    # nps.to_csv('./../data/nps/nps_train_smiles_generic_scaffold.csv',index=False)
    #
    # data = pd.read_csv('./../data/moses_train.csv')
    # data = data.sample(30000,random_state=42)
    # data['SMILES'].to_csv('./../data/moses_train_random_30000_smiles.txt',header=None,index=False)


    # zinc250k = pd.read_csv('./../data/250k_rndm_zinc_drugs_clean_3.csv')
    # zinc250k['smiles'] = zinc250k['smiles'].apply(lambda x:x.strip())
    # zinc250k['smiles'].to_csv('./../data/zinc250k.txt',index=False,header=None)

    # data = pd.read_csv('./../data/moses_train_bm_scaffold_smiles.csv')
    # data.columns = ['smiles','scaffold']
    # data['smiles'].to_csv('./../data/moses_train_bm_smiles.txt',header=False,index=False)

    # data = pd.read_csv('./../data/moses_test.csv')
    # data['scaffold'] = data['SMILES'].apply(lambda x:get_bm_scaffold(x))
    # data = data.sample(30000,random_state=0)
    # data['SMILES'].to_csv('./../data/moses_test_bm_smiles.txt',header=None,index=False)


    # data = pd.read_csv('./../model_sample/HighResNps_gen_smiles_1000_aug_100.csv',header=None)
    # data.columns = ['smiles','scaffold']
    # data['can_smiles'] = data['smiles'].apply(lambda x:Chem.MolToSmiles(Chem.MolFromSmiles(x)) if Chem.MolFromSmiles(x) else None )
    # data_uniqie_list = data['can_smiles'].unique().tolist()
    # with open('./../model_sample/unique_gen_nps.txt','w') as f:
    #     for smi in data_uniqie_list:
    #         if Chem.MolFromSmiles(smi) is not None:
    #             f.write(smi+'\n')

    # res = read_smiles('./../model_sample/unique_gen_nps.txt')
    # nps_train = read_smiles('./../data/cleaned_HighResNpsTrain.txt')
    # novelty_nps = [i for i in res if i not in nps_train]
    # with open('./../model_sample/novelty_gen_nps.txt', 'w') as f:
    #     for smi in novelty_nps:
    #             f.write(smi + '\n')

    # train = read_smiles('./../data/cleaned_HighResNpsTrain.txt')
    # test = read_smiles('./../data/cleaned_HighResNpsTest.txt')
    # out = [i for i in test if i not in train]
    # print(out)

    # data = read_smiles('./../model_sample/unique_gen_nps.txt')
    # li = []
    # for smi in data:
    #     mol = Chem.MolFromSmiles(smi)
    #     if mol is None:
    #         print(smi)
    #     else:
    #         sm = Chem.MolToSmiles(mol,isomericSmiles=False,
    #                          kekuleSmiles=False,
    #                          doRandom=False)
    #         li.append(sm)
    # with open('./../model_sample/unique_gen_nps_.txt', 'w') as f:
    #     for smi in li:
    #             f.write(smi + '\n')

    # data = pd.read_csv('./../data/test_scaffold_30000.csv')
    # data['SMILES'].to_csv('./../data/moses_test_smiles_30000.txt',header=None,index=False)

    # data = pd.read_csv('./../model_sample/ensemble_results.csv')
    # data['SMILES'].to_excel('./../model_sample/ensemble.xlsx',index=False,header=None)

    # ensemble = pd.read_csv('./../model_sample/ensemble_results.csv')
    # ensemble.columns = ['smiles','vote']
    # unique_smile = pd.read_csv("./../model_sample/unique_aug_100.txt",header=None)
    # unique_smile.columns = ['smiles']
    # ensemble.merge(unique_smile,on='smiles',how='inner')
    # print(ensemble)

    # data = pd.read_csv('./../data/2154_scaf.txt',header=None)
    # data.columns = ['smiles']
    # data['mw'] = data['smiles'].apply(lambda x: round(get_mw(x), 1))
    # # data = data.drop_duplicates(subset='mw')
    # data.to_csv('./../data/2154_scaf_mw.csv',header=None,index=False)

    # smi = 'O=C(CC)N(C1=CC=CC=C1)C(CC2)CCN2CCC3=CC=CC=C3'
    # scaf = get_mol_chain_aroma(smi)
    # print(scaf)

    smi='CN1C(=O)C(NC(=O)c2cc(Cc3ccccc3)on2)COc2ccccc21'
    scaf = get_bm_scaffold(smi)
    print(scaf)
