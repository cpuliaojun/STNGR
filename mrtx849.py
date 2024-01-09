from rdkit import Chem
from rdkit.Chem import DataStructs
import pandas as pd
from utils import read_smiles,canonic_smiles

def mol_sim(smi2):
    smi1 = 'ClC1=CC=CC2=C1C(N3CC(N=C(OCC4N(C)CCC4)N=C5N6CCN(C(C(F)=C)=O)C(CC#N)C6)=C5CC3)=CC=C2'
    mols = [Chem.MolFromSmiles(smi1),Chem.MolFromSmiles(smi2)]
    fps = [Chem.RDKFingerprint(x) for x in mols]
    sim = DataStructs.FingerprintSimilarity(fps[0],fps[1])
    return sim

def change(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    out = Chem.MolToSmiles(mol,kekuleSmiles=True)
    return out
smiles = read_smiles('./../model_sample/1000_fentanyl1000_1.0.txt')
mol = [canonic_smiles(i) for i in smiles]

mol = [i for i in mol if i is not None]
mol = [change(i) for i in mol]
# data = pd.read_csv('./../model_sample/1000_MRTX8491000_1.0.txt')
df = pd.DataFrame({'smiles':mol})
df['sim'] = df['smiles'].apply(lambda x:mol_sim(x))
df = df.sort_values(by='sim',ascending=False)
df.drop_duplicates(subset='smiles')
df.to_csv('./../data/fentanyl.csv',index=False)
# print(df[:10])