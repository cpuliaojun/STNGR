import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from utils import read_smiles,fraction_valid_smiles,canonic_smiles
import numpy as np

def get_mw(smi):
    mol = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(mol)
    return mw
# data = pd.read_csv('./../data/2154_scaf.txt',header=None)
# data.columns = ['scaffold']
# data = data.drop_duplicates(ignore_index=True)
# data['mw'] = data['scaffold'].apply(lambda x:int(get_mw(x)))
# data = data[data['mw']<=350]
# out = data.sample(100,random_state=0,ignore_index=True)
# out.to_csv('./../data/100_sort_scaffold-350.csv',index=False)

def unque_check(li):
    canon_smiles = [canonic_smiles(s) for s in li if canonic_smiles(s) is not None]
    unique_smiles = np.unique(canon_smiles)
    return unique_smiles.shape[0]/len(canon_smiles)
valid,unique = [],[]
res = read_smiles('./../model_sample/1000_sort_scaf-3501000_1.0.txt')
n=1000
chunks = [res[i:i + n] for i in range(0, len(res), n)]
for li in chunks:
    valid.append(fraction_valid_smiles(li))
    unique.append(unque_check(li))
print(valid)
print(unique)
mw = pd.read_csv('./../data/100_sort_scaffold-350.csv')['mw']
df = pd.DataFrame({'mw':mw,
                   'valid':valid,
                   'unique':unique
                   })
# print(df)
df.to_csv(
    './../data/mw_valid_unique-350.csv',index=False
)