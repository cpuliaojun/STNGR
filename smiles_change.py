from rdkit import Chem
import pandas as pd
import numpy as np

"""
用于分子两种SMILES之间的相互转换，芳香苯环和开库勒苯环
"""
def KechangeStan(smi):
    temp = Chem.MolFromSmiles(smi)
    Chem.SanitizeMol(temp)
    return Chem.MolToSmiles(temp)

def StanchangeKe(smi):
    a = Chem.MolFromSmiles(smi)
    Chem.Kekulize(a)
    return Chem.MolToSmiles(a, kekuleSmiles=True)

def stand_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    out = Chem.MolToSmiles(mol,
                           # isomericSmiles=False,
                           # kekuleSmiles=False,
                           # doRandom=False
                           )
    return out

# smi = 'O=C(C=1OC=CC1)N(C=2C=CC=CC2)C3CCN(CC=4C=CC=CC4)CC3'
# smi = 'O=C(N(C=1C=CC=CC1)C2CCN(CCC=3C=CC=CC3C)CC2)CC'  # 这个有
# smi = 'O=C(N(C=1C=CC=C(F)C1)C2CCN(CCC=3C=CC=CC3)CC2)CC'
# smi = 'O=C(C=1OC=CC1)N(C2=CC=C(OC)C=C2)C3CCN(CCC=4C=CC=CC4)CC3'
# smi = 'O=C(C=1OC=CC1)N(C=2C=CC=CC2)C3CCN(CCC=4C=CC=CC4)CC3'

# smi = 'CC(C)(C)C(C(=O)OC)NC(=O)C1=CN(C2=CC=CC=C21)CCCCCF'
# smi = 'CCOC(=O)C(C(C)C)NC(=O)C1=CN(C2=CC=CC=C21)CCCCCF'
# smi = 'O=C(NC(C)(C1=CC=CC=C1)C)C2=NN(CCCCC#N)C3=CC=CC=C32'
# smi = 'CC(C)(C)C(C(=O)OC)NC(=O)C1=CN(C2=CC=CC=C21)CCCCF'
# smi = 'CC(C)(C)C(C(=O)OC)NC(=O)C1=NN(C2=CC=CC=C21)CCCC=C'
# smi = 'CCCCN1C2=CC=CC=C2C(=N1)C(=O)NC(C(=O)N)C(C)(C)C'
# smi = 'CC(C)(C)C(C(=O)N)NC(=O)C1=NN(C2=CC=CC=C21)CCCC=C'

# smi = 'OC(CC(N(C1=CC=CC=C1)C(CC2)CCN2CCC3=CC=CC=C3)=O)=O'
# smi = 'O=C(CCC(O)=O)N(C1=CC=CC=C1)C(CC2)CCN2CCC3=CC=CC=C3'  #!!
# smi = 'O=C(CCC(NCC(OC(C)(C)C)=O)=O)N(C1=CC=CC=C1)C(CC2)CCN2CCC3=CC=CC=C3'
# smi = 'O=C(CCC(NCCC(O)=O)=O)N(C1=CC=CC=C1)C(CC2)CCN2CCC3=CC=CC=C3'
# smi = 'CCCCCn1cc(C(=O)c2cccc3cc(CC)ccc23)c2ccccc21'
# smi = 'CCCCc1ccc(C(=O)c2cn(CCCC)c3ccccc23)c2ccccc12'
# smi = 'O=C(c1cccc2ccccc12)c1cn(CCCCCCCO)c2ccccc12'
# smi = 'CCC(CC)Cn1cc(C(=O)c2cccc3ccccc23)c2ccccc21'
smi = 'CC(C)CCCn1cc(C(=O)c2cccc3ccccc23)c2ccccc21'
exa_smi = 'CN1C2=CC=CC=C2OCC(C1=O)NC(=O)C3=NOC(=C3)CC4=CC=CC=C4'


# smi = 'C(C)C1C=C(C=CC=1)C(C(NCC)C)=O'
# print(KechangeStan(smi))
print(StanchangeKe(smi))
print(KechangeStan(exa_smi))

# fentanyl = pd.read_csv('./../cal_outcome/fentanyl.csv')
# fentanyl['new_score'] = fentanyl['vote']-fentanyl['freq']-fentanyl['sa']/10
#
# fentanyl = fentanyl.sort_values(by='new_score',ascending=False)
# fentanyl['new_rank'] = np.array(range(1,fentanyl.shape[0]+1))
# fentanyl.to_csv('./../cal_outcome/new_fentanyl.csv',index=False)
# print(StanchangeKe(smi))
# print(stand_smi(smi))
#
# sorted_res = pd.read_csv('./../cal_outcome/frequency_results.csv')
# sorted_res['can_smiles'] = sorted_res['SMILES'].apply(lambda x:stand_smi(x))
# sorted_res.to_csv('./../cal_outcome/novel_can_results.csv',index=False)