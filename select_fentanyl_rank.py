import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs,Descriptors,Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import matplotlib.pyplot as plt
import numpy as np

def get_bm_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    bm = MurckoScaffold.GetScaffoldForMol(mol)
    chain = Chem.MolToSmiles(bm,
                             isomericSmiles=False,
                             kekuleSmiles=False,
                             doRandom=False)
    return chain


data = pd.read_csv('./../cal_outcome/final_all.csv')
del data['index']
# print(data)

data['scaffold'] = data['SMILES'].apply(lambda x:get_bm_scaffold(x))
fentanyl = data[data['scaffold']=='c1ccc(CCN2CCC(Nc3ccccc3)CC2)cc1']
fentanyl['rank'] = np.array(range(1,fentanyl.shape[0]+1))
fentanyl.to_csv('./../cal_outcome/fentanyl.csv',index=False)
print(fentanyl)

fentanyl_smi = "O=C(O)CCC(=O)N(c1ccccc1)C1CCN(CCc2ccccc2)CC1"

fentanyl_scaf = get_bm_scaffold(fentanyl_smi)
print(fentanyl_scaf)


template = Chem.MolFromSmiles('c1ccc(CCN2CCC(Nc3ccccc3)CC2)cc1')
AllChem.Compute2DCoords(template)
# smis = [
#     'COc1c(OC)cc(C(C2CCCCC2)O)cc1',
#     'Cc1c(Br)cc(C(C2C(O)CCCC2)O)cc1',
#     'CN(C(NC1C(Cc2ccccc2)CCCC1)=O)C',
#     'CN(C(CCNC(NC(C1CCCCC1)c1c(F)cccc1)=O)=O)C',
#     'CN(S(=O)(=O)C)CC(NC(c1c(CC2CCCCC2)cccc1)(CO)C)=O'
# ]
smis = [fentanyl_scaf,fentanyl_smi]
mols = []
for smi in smis:
    mol = Chem.MolFromSmiles(smi)
    # 生成一个分子的描述，其中一部分 分子被约束为具有与参考相同的坐标。
    AllChem.GenerateDepictionMatching2DStructure(mol, template)
    mols.append(mol)

# 基于分子文件输出分子结构
img = Draw.MolsToGridImage(
    mols,  # mol对象
    molsPerRow=2,
    subImgSize=(200, 200),
    legends=['' for x in mols]
)
# img.save('./../pic/scaffold_molecule.png')
plt.imshow(img)
plt.show()

