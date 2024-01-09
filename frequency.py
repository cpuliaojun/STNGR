import pandas as pd
from tqdm import tqdm
from utils import read_smiles,canonic_smiles,write_smiles
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors,AllChem,DataStructs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# sns.palplot(sns.color_palette("Pastel1"))
# sns.set_style("whitegrid")
sns.color_palette('Blues')
# sns.color_palette(palette='Blues_r')
data = read_smiles('./../model_sample/nps_aug_withoutVC1000_1.0.txt')
# data = read_smiles('./../model_sample/nps_tl_withoutVC1000_1.0.txt')
train = read_smiles('./../data/nps/cleaned_HighResNpsTrain.txt')
def exact_mass_from_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    mass = Descriptors.ExactMolWt(mol)
    return mass

def sim(smi1,smi2):
    """

    Args:
        smi1: 第一个smiles
        smi2: 第二个smiles

    Returns: 两个smiles对应的分子的相似度

    """
    try:
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        mols = [mol1,mol2]
        fps = [AllChem.GetMorganFingerprint(x, radius=2) for x in mols]
        sim = DataStructs.DiceSimilarity(fps[0], fps[1])
    except:
        sim = 0
    return sim

def max_tanimoto(smi):
    """

    Args:
        smi: 给定的SMILES

    Returns: 给定的SMILES和训练数据中最相似的分子的相似度

    """
    train = read_smiles('./../data/nps/cleaned_HighResNpsTrain.txt')
    sim_li = [sim(smi,train_item) for train_item in train]
    out = np.max(sim_li)
    # print(out)
    return out


res = dict()
canon = [canonic_smiles(s) for s in tqdm(data) if canonic_smiles(s) is not None]  # 把每个SMILES分子做规范化处理
# novelty = [i for i in tqdm(canon) if i not in train]
# write_smiles(novelty,'./../cal_outcome/tl_novelty_molecules.txt')
#计算投票结果
for smi in canon:
    if smi not in res:
        res[smi] = 1
    else:
        res[smi]+=1
sorted_res = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
save_path = './../cal_outcome/'
with open(f'{save_path}frequency_results.csv', 'w') as f:
    w = csv.writer(f)
    w.writerow(['SMILES', 'freq'])
    for x in sorted_res:
        w.writerow([x[0], x[1]])
sorted_res = pd.read_csv('./../cal_outcome/frequency_results.csv')
# 计算每一个SMILES出现的次数，然后降序排列，然后输出到一个csv文件
# sorted_res = pd.read_csv('./../cal_outcome/sorted_res.csv')
sorted_res.columns = ['smiles','num']
sorted_res['max_tanimoto'] = sorted_res['smiles'].apply(lambda x:max_tanimoto(x))
sorted_res.to_csv('./../cal_outcome/sorted_res.csv',index=False)
sorted_res['smiles'].to_csv('./../model_sample/aug_novelty_molecules.txt',header=None,index=False)
print(sorted_res.iloc[:20,1:])

frequency_tc = sorted_res.iloc[:,1:]
frequency_tc = frequency_tc[frequency_tc['max_tanimoto']<1]

def get_group(num):
    if num == 1:
        group = 1
    elif 1 < num <= 3:
        group = 3
    elif 3 < num <= 5:
        group = 5
    elif 5 < num <= 10:
        group = 10
    elif 10<num <= 30:
        group = 30
    elif 30 <num <= 50:
        group = 50
    elif 50 < num <= 100:
        group = 100
    elif 100< num <=200:
        group = 200
    elif 200< num <=400:
        group = 400
    elif 400< num:
        group = 500
    else:
        group = 0
    return group
frequency_tc['group'] = frequency_tc['num'].apply(lambda x:get_group(x))
frequency_tc = frequency_tc[frequency_tc['group'] != 0]


res_groupby = sorted_res.groupby(['num']).count()
res = res_groupby.reset_index()
print(res_groupby)
print(res)
#
fig = plt.figure(figsize=(16,6))
fig.add_subplot(121)
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Sample frequency')
plt.ylabel('Count of unique molecules')
plt.scatter(x=res['num'],y=res['smiles'],s=10)
fig.add_subplot(122)
sns.boxplot(x="group", y="max_tanimoto", data=frequency_tc)
plt.ylabel('NN_Tc')
plt.savefig('./../pic/Tc_frequency.png',dpi=600,pad_inches=0.0)
plt.show()

