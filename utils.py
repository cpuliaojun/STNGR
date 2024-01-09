# -*- encoding: utf-8 -*-
"""
@file         :    utils.py   
@description  :
@date         :    2022/8/11 17:25
@author       :    silentpotato  
"""
import re
from rdkit import Chem
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import rdkit.Chem.QED as QED
from outmodel import sascorer, npscorer
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit import DataStructs


from rdkit import rdBase
rdBase.DisableLog('rdApp.error')

def write_smiles(smiles, smiles_file):
    '''

    Args:
        smiles: SMILES字符串列表
        smiles_file: 需要写入的文件名称

    Returns:
        将SMILES字符串列表中的每一个字符串写入文件中
    '''
    # write sampled SMILES
    with open(smiles_file, 'w') as f:
        for sm in smiles:
            _ = f.write(sm + '\n')


def read_smiles(smiles_file) -> [str]:
    '''

    Args:
        smiles_file:每一行是一个SMILES字符串的文件

    Returns:
        smiles：字符串列表
    '''
    smiles = []
    with open(smiles_file, 'r') as f:
        smiles.extend([line.strip() for line in f.readlines() \
                       if line.strip()])
    return smiles


def get_max_lenth(smiles):
    '''

    Args:
        smiles: SMILES字符串的列表

    Returns:
        SMILES字符串列表中长度最长的SMILES字符串的长度
    '''
    pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    # lens = []
    # for i in smiles:
    #     # regex = re.compile(pattern)
    #     try:
    #         len = regex.findall(i.strip())
    #     except:
    #         print(i)
    #     lens.append(len)
    lens = [len(regex.findall(i.strip())) for i in smiles]
    smiles_max_len = max(lens)
    return smiles_max_len
    # def len(smi):
    #     REGEXPS = {
    #         "brackets": re.compile(r"(\[[^\]]*\])"),
    #         "2_ring_nums": re.compile(r"(%\d{2})"),
    #         "brcl": re.compile(r"(Br|Cl)")
    #     }
    #     REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]
    #     def split_by(smi, regexps):
    #         regexp = REGEXPS[regexps[0]]
    #         splitted = regexp.split(smi)
    #         tokens = []
    #         for i, split in enumerate(splitted):
    #             if i % 2 == 0:
    #                 tokens += split_by(split, regexps[1:])
    #             else:
    #                 tokens.append(split)
    #         return tokens
    #
    #     tokens_without_padding = split_by(smi,REGEXP_ORDER)
    #     return len(tokens_without_padding)
    # lens = [len(smi) for smi in smiles]
    # return max(lens)




def check_novelty(gen_smiles, train_smiles):  # gen: say 788, train: 120803
    '''

    Args:
        gen_smiles: 生成SMILES字符串的列表
        train_smiles: 训练的SMILES字符串的列表

    Returns:
        生成出的不在训练集中的SMILES字符串的比例

    '''
    if len(gen_smiles) == 0:
        novel_ratio = 0.
    else:
        duplicates = [1 for mol in gen_smiles if mol in train_smiles]  # [1]*45
        novel = len(gen_smiles) - sum(duplicates)  # 788-45=743
        # novel_ratio = novel*100./len(gen_smiles)  # 743*100/788=94.289
        novel_ratio = novel / len(gen_smiles)  # 743*100/788=94.289
    # print("novelty: {:.3f}%".format(novel_ratio))
    return novel_ratio


def check_in(gen_smiles, test_unique_smiles):
    '''

    Args:
        gen_smiles: 生成SMILES字符串的列表
        test_unique_smiles: 测试独特SMILES的列表

    Returns:
        生成的SMILES的独特性指标
    '''
    duplicates = [mol for mol in test_unique_smiles if mol in gen_smiles]
    not_in = [mol for mol in test_unique_smiles if mol not in gen_smiles]
    return len(duplicates) / len(test_unique_smiles), duplicates, not_in


def canonic_smiles(smiles_or_mol):
    '''

    Args:
        smiles_or_mol: 一个SMILES字符串或Mol对象

    Returns:
        一个SMILES字符串
    '''
    mol = get_mol(smiles_or_mol)
    if mol is None:
        return None
    if mol is not None:
        return Chem.MolToSmiles(mol)


def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol

def seqs_to_smiles(seqs,vocabulary):
    smiles = []
    for seq in seqs:
        smi = vocabulary.decode(seq)
        smiles.append(smi)
    return smiles


def clean_mol(smiles, stereochem=False):
    """
    Construct a molecule from a SMILES string, removing stereochemistry and
    explicit hydrogens, and setting aromaticity.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("invalid SMILES: " + str(smiles))
    if not stereochem:
        # 去除立体化学结构信息
        Chem.RemoveStereochemistry(mol)
    Chem.SanitizeMol(mol)
    mol = Chem.RemoveHs(mol)
    return mol


def clean_mols(all_smiles, stereochem=False, selfies=False, deepsmiles=False):
    """
    Construct a list of molecules from a list of SMILES strings, replacing
    invalid molecules with None in the list.
    """
    mols = []
    for smiles in tqdm(all_smiles):
        try:
            mol = clean_mol(smiles, stereochem)
            mols.append(mol)
        except ValueError:
            mols.append(None)
    return mols
def plot_loss(file):
    res = pd.read_csv(file)
    fig,axes = plt.subplots()
    axes.plot(res['epoch'],res['train_loss'],label='Train loss')
    axes.plot(res['epoch'],res['val_loss'],label='Val loss')
    axes.legend()
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.set_title("Pretrain")

    plt.savefig('./../pic/pretrain_loss.png',dpi=300)
    # plt.show()

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).to(device)
    return torch.autograd.Variable(tensor)

def unique(arr):
    # Finds unique rows in arr and return their indices
    arr = arr.cpu().numpy()
    arr_ = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize * arr.shape[1])))
    _, idxs = np.unique(arr_, return_index=True)
    if torch.cuda.is_available():
        return torch.LongTensor(np.sort(idxs)).cuda()
    return torch.LongTensor(np.sort(idxs))

class qed_func():
    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(0)
            else:
                try:
                    qed = QED.qed(mol)
                except:
                    qed = 0
                scores.append(qed)
        return np.float32(scores)

class sa_func():

    def __call__(self, smiles_list):
        scores = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                scores.append(100)
            else:
                scores.append(sascorer.calculateScore(mol))
        return np.float32(scores)

def fraction_valid_smiles(smiles):
    """Takes a list of SMILES and returns fraction valid."""
    i = 0
    for smile in smiles:
        if Chem.MolFromSmiles(smile):
            i += 1
    return i / len(smiles)

def to_tensor(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)




def get_fp(smiles_file):
    smiles = read_smiles(smiles_file)
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    fingerprints = []
    safe = []
    for mol_idx, mol in enumerate(mols):
        try:
            fingerprint = [x for x in MACCSkeys.GenMACCSKeys(mol)]
            fingerprints.append(fingerprint)
            safe.append(mol_idx)
        except:
            print("Error", mol_idx)
            continue
    fp_MACCSkeys = pd.DataFrame(fingerprints)
    # print(fp_MACCSkeys)
    return fp_MACCSkeys

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

def filter(smiles_li):
    valid_smiles = []
    for smi in smiles_li:
        if Chem.MolFromSmiles(smi):
            valid_smiles.append(smi)
    return valid_smiles,len(valid_smiles)


if __name__ == "__main__":
    # plot_loss('./../logs/pretrain_712301_embd_128_nhead8_nhid512_log.csv')
    smi='CCC1=CC(=CC=C1)C(=O)C(C)NCC'
    clean_smi=canonic_smiles(smi)
    print(clean_smi)