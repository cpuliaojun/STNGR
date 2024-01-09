import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from utils import read_smiles,write_smiles

def get_generic_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    chain = Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol),
                             isomericSmiles=False,
                             kekuleSmiles=False,
                             doRandom=False)
    return chain

def get_bm_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    bm = MurckoScaffold.GetScaffoldForMol(mol)
    chain = Chem.MolToSmiles(bm,
                             isomericSmiles=False,
                             kekuleSmiles=False,
                             doRandom=False)
    return chain

# test = pd.read_csv('./../data/moses_raw/test.csv')
# test_30000 = test.sample(30000,random_state=42)
# test_30000['generic_scaffold'] = test_30000['SMILES'].apply(lambda x:get_generic_scaffold(x))
# test_30000['bm_scaffold'] = test_30000['SMILES'].apply(lambda x:get_bm_scaffold(x))
# test_30000.to_csv('./../data/test_30000.csv',index=False)
#
#
#
# test_scaffold = pd.read_csv('./../data/moses_raw/test_scaffolds.csv')
# test_scaffold_30000 = test_scaffold.sample(30000,random_state=42)
# test_scaffold_30000['generic_scaffold'] = test_scaffold_30000['SMILES'].apply(lambda x:get_generic_scaffold(x))
# # test_scaffold_30000['bm_scaffold'] = test_30000['SMILES'].apply(lambda x:get_bm_scaffold(x))
# test_scaffold_30000.to_csv('./../data/test_scaffold_30000.csv',index=False)


# test_30000 = pd.read_csv('./../data/test_30000.csv')
# test_scaffold_30000 = pd.read_csv('./../data/test_scaffold_30000.csv')
# test_30000['SMILES'].to_csv('/share/nps/Sc2Mol/data/eval.txt',header=None,index=False)
# test_scaffold_30000['SMILES'].to_csv('/share/nps/Sc2Mol/data/eval_scaffold.txt',header=None,index=False)


all_2023 = read_smiles('./../data/nps/out_all_2023.txt')
train = read_smiles('./../data/nps/cleaned_HighResNpsTrain.txt')
test = [i for i in all_2023 if i not in train]
print(len(test))
print(test)
write_smiles(test,'./../data/nps/2023_test_nps.txt')