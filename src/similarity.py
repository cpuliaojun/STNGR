from rdkit import Chem
from rdkit.Chem import DataStructs,AllChem
import pandas as pd
from utils import read_smiles
from rdkit.Chem.Scaffolds import MurckoScaffold

def mol_sim(smi1,smi2):
    mols = [Chem.MolFromSmiles(smi1),Chem.MolFromSmiles(smi2)]
    # fps = [Chem.RDKFingerprint(x) for x in mols]
    fps = [AllChem.GetMorganFingerprint(x, radius=2) for x in mols]
    sim = DataStructs.FingerprintSimilarity(fps[0],fps[1])
    return sim

def get_mol_chain_aroma(smi):
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



def sim(smi1,smi2):
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        mols = [mol1,mol2]
        fps = [AllChem.GetMorganFingerprint(x, radius=2) for x in mols]
        sim = DataStructs.DiceSimilarity(fps[0], fps[1])
    except:
        sim = 0
    return sim

def scaffold_sim(smi1,smi2):
    try:
        # scaf1_mol = get_mol_chain_aroma(smi1)
        scaf1_mol = get_bm_scaffold(smi1)
        scaf1 = Chem.MolFromSmiles(scaf1_mol)
        # scaf2 = get_mol_chain_aroma(smi2)
        scaf2 = Chem.MolFromSmiles(smi2)
        mols = [scaf1, scaf2]
        fps = [AllChem.GetMorganFingerprint(x, radius=2) for x in mols]
        sim = DataStructs.DiceSimilarity(fps[0], fps[1])
    except:
        sim = 0
    return sim


gen_smiles = read_smiles('./../model_sample/bm_test_scaffold1.txt')
test_smiles = pd.read_csv('./../data/test_scaffold_30000.csv')['SMILES'].to_list()
count = 0
sim_li = []
for gen,test in zip(gen_smiles,test_smiles):
    si = scaffold_sim(gen,test)
    if si !=0:
        count += si
        sim_li.append(si)
result = count / len(sim_li)
print(result)


