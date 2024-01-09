import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs,Descriptors,Draw
from rdkit import rdBase
from rdkit.Chem.Draw.MolDrawing import  MolDrawing,DrawingOptions
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import read_smiles,canonic_smiles
import seaborn as sns
from tqdm import tqdm
from umap import UMAP
# from src.get_scaffold import get_mol_chain_aroma,get_mw
import pickle

rdBase.DisableLog('rdApp.error')

class FP:
    """
    建立一个FP的类方便后续的分子指纹处理
    """
    def __init__(self, fp, names):
        self.fp = fp
        self.names = names
    def __str__(self):
        return "%d bit FP" % len(self.fp)
    def __len__(self):
        return len(self.fp)

def get_cfps(mol, radius=2, nBits=1024, useFeatures=False, counts=False, dtype=np.float32):
    arr = np.zeros((1,), dtype)
    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures, bitInfo=info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return FP(arr, range(nBits))

def calFP(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        fp = get_cfps(mol)
        return fp
    except Exception as e:
        return None

# def get_valid_and_unique(li):
#     valid_li = []
#     for smi in li:
#         mol = Chem.MolFromSmiles(smi)
#         if mol:
#             valid_li.append(smi)
#     valid = len(valid_li)/len(li)
#     unique = len(list(set(valid_li)))/len(valid_li)
#     return valid,unique


def plot_tsne():
    # raw = pd.read_csv('./../data/moses_train_scaffold.csv')
    raw = pd.read_csv('./../data/nps/cleaned_HighResNpsTrain.csv')
    # raw.columns = ['smiles','scaffold']
    pretrain_data = raw.sample(2154,random_state=0, replace=True)
    pretrain_data['fp'] = pretrain_data['clean_SMILES'].apply(calFP)

    gen = pd.read_csv('./../cal_outcome/sorted_res.csv')
    # gen = gen.sample(5000)
    # gen.columns = ['smiles']
    gen['fp'] = gen['smiles'].apply(calFP)
    print(len(gen))
    # gen = gen[gen['fp'] != None]
    gen = gen.dropna(subset='fp')
    print(len(gen))
    gen = gen.sample(2154)

    data = pd.concat([pretrain_data['fp'],gen['fp']],axis=0)
    # data.to_csv('./../data/plot_tsne.csv',header=None,index=False)
    data = pd.DataFrame(data, columns=['fp'])

    # tsne_model = TSNE(n_components=2, random_state=42, perplexity=100, n_iter=500)
    tsne_model = UMAP(n_components=2,
                      n_neighbors=15,
                      random_state=0,
                      metric='euclidean',
                      min_dist=0.6)

    # pretrain_x = np.array([x.fp for x in pretrain_data['fp']])
    # gen_x = np.array([x.fp for x in gen['fp']])
    #
    # pretrain_tsne_result = tsne_model.fit_transform(pretrain_x)
    # gen_tsne_result = tsne_model.fit_transform(gen_x)
    #
    # pretrain_data['tsne_1'] = pretrain_tsne_result.T[0]
    # pretrain_data['tsne_2'] = pretrain_tsne_result.T[1]
    # gen['tsne_1'] = gen_tsne_result.T[0]
    # gen['tsne_2'] = gen_tsne_result.T[1]
    # a=0
    # for i in data['fp']:
    #     try:
    #         a += 1
    #         print(i.fp)
    #     except:
    #         print(i)
    #         print(a)
    #         break


    x = np.array([i.fp for i in data['fp']])
    res = tsne_model.fit_transform(x)
    data['tsne1'] = res.T[0]
    data['tsne2'] = res.T[1]
    data.to_csv('./../cal_outcome/umap_tem15.csv')



    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(pretrain_data['tsne_1'], pretrain_data['tsne_2'], s=5, alpha=0.5, label='pretrian data')
    # ax.scatter(gen['tsne_1'], gen['tsne_2'], s=5, alpha=0.5, label='pretrian data')
    ax.scatter(data['tsne1'].iloc[:2154],data['tsne2'].iloc[:2154],s=5,alpha=0.5,label='Known NPS')
    ax.scatter(data['tsne1'].iloc[2154:],data['tsne2'].iloc[2154:],s=5,alpha=0.5,label='Generated')


    ax.set_xlabel('UMAP_1')
    ax.set_ylabel('UMAP_2')
    ax.legend()
    # plt.savefig('./../pic/tsne_200.png', dpi=200)
    plt.savefig('./../pic/umap_tem15.png', dpi=200)
    plt.show()

def get_valid_and_unique(li):
    canon_smiles = [canonic_smiles(s) for s in tqdm(li) if canonic_smiles(s) is not None]
    unique_smiles = np.unique(canon_smiles)
    valid = len(canon_smiles)/len(li)
    unique = unique_smiles.shape[0]/len(li)
    return valid ,unique

def get_mw(smi):
    mol = Chem.MolFromSmiles(smi)
    mw = Descriptors.MolWt(mol)
    return mw

def get_numAtom(smi):
    mol = Chem.MolFromSmiles(smi)
    numAtom = mol.GetNumAtoms()
    return numAtom

def plot_mw_valid_unique():
    # sample = read_smiles('./../model_sample/moses_test_bm_scaffold_100.txt')
    # valid_li,unique_li= [],[]
    # for idx in range(0,100000,100):
    #     temp = sample[idx:idx+100]
    #     valid,unique = get_valid_and_unique(temp)
    #     valid_li.append(valid)
    #     unique_li.append(unique)
    # print(valid_li)
    # print(unique_li)
    # df = pd.DataFrame({'validity':valid_li,
    #                    'uniqueness':unique_li
    #                    })
    # scaffold = pd.read_csv('./../data/moses_test.csv').sample(1000,random_state=0)
    # scaffold['mw'] = scaffold['SMILES'].apply(lambda x:get_mw(x))
    # mw = pd.DataFrame(scaffold['mw'].values,columns=['mw'])
    # # df = pd.read_csv('./../data/plot_df.csv')
    # plot_df = pd.concat([mw,df],axis=1)
    # # plot_df.to_csv('./../data/plot_df.csv')
    plot_df = pd.read_csv('./../data/mw_valid_unique.csv')
    plot_df = plot_df[plot_df['mw']<= 350]
    fig = plt.figure(figsize=(8,6))
    # fig.add_subplot(111)
    # sns.jointplot(x='mw',y='valid',data=plot_df)
    # sns.jointplot(x='mw',y='unique',data=plot_df)
    # sns.regplot(x='mw',y='valid',data=plot_df)
    # plt.xlabel('MW')
    # plt.ylabel('Validity')

    fig.add_subplot(111)
    sns.regplot(x='mw',y='unique',data=plot_df)
    plt.xlabel('MW')
    plt.ylabel('Uniqueness')



    # sns.pairplot(plot_df)
    plt.savefig('./../pic/mw_uniqueness.png',dpi=600)
    plt.show()

    # 相关性检验
    import numpy as np
    import scipy.stats as stats
    import scipy

    valid_cor,valid_p = stats.stats.pearsonr(plot_df['mw'],plot_df['valid'])
    unique_cor,unique_p = stats.stats.pearsonr(plot_df['mw'],plot_df['unique'])
    print(valid_cor,valid_p)
    print(unique_cor,unique_p)

    # pretrain= pd.read_csv('./../data/pretrain_712301.txt')
    # pretrain.columns = ['smiles','bm_scaffold']
    # pretrain_1000 = pretrain.sample(1000,random_state=0)
    # pretrain_1000['generic_scaffold'] = pretrain_1000['smiles'].apply(lambda x:get_mol_chain_aroma(x))
    # pretrain_1000['scaffold_mw'] = pretrain_1000['generic_scaffold'].apply(lambda x:get_mw(x))
    # pretrain_1000 = pretrain_1000.sort_values(by='scaffold_mw')
    # print(pretrain_1000)
    # pretrain_1000['generic_scaffold'].to_csv('./../data/valid_unque_mw_chembl_1000_generic_scaffold.txt',header=False,index=False)

def plot_distribute():
    # kernel density estimate (KDE) plot
    # font_options = {
    #     'family' : 'serif', # 设置字体家族
    #     'serif' : 'simsun', # 设置字体
    # }
    # plt.rc('font',**font_options)

    # res = pd.read_csv('./../cal_outcome/test_30000_generic_scaffold_withoutVC1-outcomes.csv')
    # ref = pd.read_csv('./../cal_outcome/moses_train_random_30000_smiles-outcomes.csv')
    # res = pd.read_csv('./../cal_outcome/probility_sampled-outcomes.csv')
    res = pd.read_csv('./../cal_outcome/sorted_res-outcomes.csv')
    ref = pd.read_csv('./../cal_outcome/cleaned_HighResNpsTrain-outcomes.csv')
    # print(res.head())
    # print(ref.head())
    plt.figure(figsize=(16,12))
    """
    Density Plot of SA
    """
    plt.subplot(3,3,1)
    sns.kdeplot(res.loc[res['outcome'] == 'Synthetic accessibility score', 'value'], shade=True, alpha=0.5, label='Ours')
    sns.kdeplot(ref.loc[ref['outcome'] == 'Synthetic accessibility score', 'value'], shade=True, alpha=0.5, label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol['outcome'] == 'Synthetic accessibility score', 'value'], shade=True, alpha=0.5, label='Sc2Mol')
    plt.xlabel('SA score')
    plt.ylabel('Density')
    # plt.title('Density Plot of SA score')
    plt.legend()

    """
    Density Plot of LogP
    """
    plt.subplot(332)
    sns.kdeplot(res.loc[res['outcome'] == 'LogP', 'value'], shade=True, alpha=0.5, label='Ours')
    sns.kdeplot(ref.loc[ref['outcome'] == 'LogP', 'value'], shade=True, alpha=0.5, label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol['outcome'] == 'LogP', 'value'], shade=True, alpha=0.5, label='Sc2Mol')
    plt.xlabel('LogP')
    plt.ylabel('Density')
    # plt.title('Density Plot of LogP')
    plt.legend()

    """
    Density Plot of NP
    """
    plt.subplot(333)
    sns.kdeplot(res.loc[res['outcome']=="Natural product-likeness score","value"],shade=True,alpha=0.5,label='Ours')
    sns.kdeplot(ref.loc[ref['outcome']=="Natural product-likeness score","value"],shade=True,alpha=0.5,label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol['outcome']=="Natural product-likeness score","value"],shade=True,alpha=0.5,label='Sc2Mol')
    plt.xlabel('NP score')
    plt.ylabel('Density')
    # plt.title("Density Plot of NP score")
    plt.legend()

    """
    Denstity Plot of TPSA
    """
    plt.subplot(336)
    sns.kdeplot(res.loc[res['outcome']=="TPSA","value"],shade=True,alpha=0.5,label='Ours')
    sns.kdeplot(ref.loc[ref['outcome']=="TPSA","value"],shade=True,alpha=0.5,label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol['outcome']=="TPSA","value"],shade=True,alpha=0.5,label='Sc2Mol')
    plt.xlabel('TPSA')
    plt.ylabel('Density')
    # plt.title("Density Plot of TPSA")
    plt.legend()

    plt.subplot(335)
    sns.kdeplot(res.loc[res["outcome"]=='Molecular weight',"value"],shade=True,alpha=0.5,label='Ours')
    sns.kdeplot(ref.loc[ref["outcome"]=='Molecular weight',"value"],shade=True,alpha=0.5,label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol["outcome"]=='Molecular weight',"value"],shade=True,alpha=0.5,label='Sc2Mol')
    plt.xlabel('Molecular weight')
    plt.ylabel('Density')
    # plt.title("Density Plot of Molecular weigh")
    plt.legend()

    plt.subplot(334)
    sns.kdeplot(res.loc[res["outcome"]=='BertzTC',"value"],shade=True,alpha=0.5,label='Ours')
    sns.kdeplot(ref.loc[ref["outcome"]=='BertzTC',"value"],shade=True,alpha=0.5,label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol["outcome"]=='BertzTC',"value"],shade=True,alpha=0.5,label='Sc2Mol')
    plt.xlabel('BertzTC')
    plt.ylabel('Density')
    # plt.title("Density Plot of BertzTC")
    plt.legend()


    plt.subplot(337)
    sns.kdeplot(res.loc[res["outcome"]=='QED',"value"],shade=True,alpha=0.5,label='Ours')
    sns.kdeplot(ref.loc[ref["outcome"]=='QED',"value"],shade=True,alpha=0.5,label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol["outcome"]=='QED',"value"],shade=True,alpha=0.5,label='Sc2Mol')
    plt.xlabel('QED')
    plt.ylabel('Density')
    # plt.title("Density Plot of QED")
    plt.legend()

    plt.subplot(338)
    sns.kdeplot(res.loc[res["outcome"]=='% sp3 carbons',"value"],shade=True,alpha=0.5,label='Ours')
    sns.kdeplot(ref.loc[ref["outcome"]=='% sp3 carbons',"value"],shade=True,alpha=0.5,label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol["outcome"]=='% sp3 carbons',"value"],shade=True,alpha=0.5,label='Sc2Mol')
    plt.xlabel('Precent sp3 carbons')
    plt.ylabel('Density')
    # plt.title("Density Plot of % sp3 carbons")
    plt.legend()

    plt.subplot(339)
    sns.kdeplot(res.loc[res["outcome"]=='# of rings',"value"],shade=True,alpha=0.5,label='Ours')
    sns.kdeplot(ref.loc[ref["outcome"]=='# of rings',"value"],shade=True,alpha=0.5,label='Training')
    # sns.kdeplot(sc2mol.loc[sc2mol["outcome"]=='# of rings',"value"],shade=True,alpha=0.5,label='Sc2Mol')
    plt.xlabel('Aromatic rings')
    plt.ylabel('Density')
    # plt.title("Density Plot of # of rings")
    plt.legend()
    # plt.savefig('./../pic/moses_sampled_distribution_300.png',dpi=300)
    plt.show()

# def plot_molecule():
#     smi= 'CN1(C2(CN)CCCCC2)CCCCC1'
#     scaf = 'CCCC(C)C1CCC2CC(CC(C)CC)CC2C1'
#     mol = Chem.MolFromSmiles(smi)
#     scaffold = get_mol_chain_aroma(smi)
#     print(scaffold)
#
#     scaffold = Chem.MolFromSmiles(scaffold)
#     # fig = plt.figure(figsize=(16,12))
#     plt.axis('off')
#     opt = DrawingOptions()
#     # opt.includeAtomNumbers=True
#     # opt.bondLineWidth=2.8
#     mol_img = Draw.MolToImage(mol)
#     scaf_img = Draw.MolToImage(scaffold)
#     plt.subplot(121)
#     plt.imshow(scaf_img)
#     plt.subplot(122)
#     plt.imshow(mol_img)
#     plt.show()

def plot_mw_NoAtom():
    nps = pd.read_csv('./../data/cleaned_HighResNpsTrain.txt')
    nps.columns = ['smiles']
    nps['Molecule Weight'] = nps['smiles'].apply(lambda x:int(get_mw(x)))
    nps['Number of Atoms'] = nps['smiles'].apply(lambda x:get_numAtom(x))
    fig = plt.figure(figsize=(14,6))
    fig.add_subplot(121)
    sns.histplot(x=nps['Molecule Weight'],kde=False)
    fig.add_subplot(122)
    sns.histplot(x=nps['Number of Atoms'])
    # plt.savefig('./../pic/train_data_count.png',dpi=600)
    plt.show()


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
    smis = pd.read_csv('./../cal_outcome/fentanyl.csv')['SMILES'].iloc[:50].to_list()
    mols = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        # 生成一个分子的描述，其中一部分 分子被约束为具有与参考相同的坐标。
        AllChem.GenerateDepictionMatching2DStructure(mol, template)
        mols.append(mol)

    # 基于分子文件输出分子结构
    img = Draw.MolsToGridImage(
        mols,  # mol对象
        molsPerRow=5,
        subImgSize=(400, 400),
        # legends=['' for x in mols]
        legends=[i for i in range(1,51)]
    )
    # img.save('./../pic/scaffold_molecule.png')
    plt.imshow(img)
    plt.show()
# def plot_fentanyl():


def plot_loss():
    data = './../data/loss/moses_general_scaffold.pkl'
    with open(data,'rb') as f:
        df = pickle.load(f)
    print(df)
    df = df.iloc[:100,:]
    fig =plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(df['epoch'],df['train_loss'])
    ax.plot(df['epoch'],df['val_loss'])
    plt.show()

if __name__ == "__main__":
    # pass
    # plot_mw_valid_unique()
    # plot_distribute()
    plot_tsne()
    # plot_molecule()
    # plot_mw_NoAtom()
    # plot_scaffold_molecule()
    # plot_loss()