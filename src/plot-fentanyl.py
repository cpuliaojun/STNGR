import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs,Descriptors,Draw
import matplotlib.pyplot as plt



def plot_scaffold_molecule():
    # template = Chem.MolFromSmiles('c1ccc(CC2CCCCC2)cc1')
    template = Chem.MolFromSmiles('c1ccc(CCN2CCC(Nc3ccccc3)CC2)cc1')
    AllChem.Compute2DCoords(template)
    # smis = [
    #     'COc1c(OC)cc(C(C2CCCCC2)O)cc1',
    #     'Cc1c(Br)cc(C(C2C(O)CCCC2)O)cc1',
    #     'CN(C(NC1C(Cc2ccccc2)CCCC1)=O)C',
    #     'CN(C(CCNC(NC(C1CCCCC1)c1c(F)cccc1)=O)=O)C',
    #     'CN(S(=O)(=O)C)CC(NC(c1c(CC2CCCCC2)cccc1)(CO)C)=O'
    # ]
    smis = pd.read_csv('./../cal_outcome/fentanyl.csv')['SMILES'].iloc[:2].to_list()
    # smis = list(pd.read_csv('./../cal_outcome/fentanyl.csv')['SMILES'].iloc[2])
    mols = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        # 生成一个分子的描述，其中一部分 分子被约束为具有与参考相同的坐标。
        AllChem.GenerateDepictionMatching2DStructure(mol, template)
        mols.append(mol)

    # 基于分子文件输出分子结构
    # plt.figure(figsize=(5,15))
    img = Draw.MolsToGridImage(
        mols,  # mol对象
        molsPerRow=1,
        subImgSize=(1000, 1000),
        # legends=['' for x in mols]
        # legends=[i for i in range(1,51)]
    )
    # img.save('./../pic/scaffold_molecule.png')
    plt.imshow(img)
    plt.show()
if __name__ == "__main__":
    plot_scaffold_molecule()
    print('he')