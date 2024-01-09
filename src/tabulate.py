# -*- encoding: utf-8 -*-
"""
@file         :    tabulate.py   
@description  :
@date         :    2022/6/28 15:50
@author       :    silentpotato

计算smiles.txt文件中分子生成指标，包括有效性，唯一性，新颖性和外部测试集中非覆盖率
"""
import os
import time
import pandas as pd
import numpy as np
from utils import check_novelty, canonic_smiles, check_in
from rdkit import RDLogger
import argparse
from tqdm import tqdm
from utils import write_smiles
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser(
    description='Calculate a series of properties for a set of SMILES')
parser.add_argument('--smiles_file', type=str,
                    help='file containing SMILES', default='./../model_sample/test_scaffold_30000_generic_scaffold_1_withVC.txt')
                    # help='file containing SMILES', default='./../data/cleaned_HighResNpsTest.txt')
parser.add_argument('--reference_file', type=str,
                    help='file containing a reference set of SMILES', default='./../data/moses_train.csv')
parser.add_argument('--test_file', type=str,default='./../data/nps/2023_test_nps.txt')
parser.add_argument('--output_dir', type=str,
                    help='directory to save output to', default='./../tabulate')
args = parser.parse_args()

filename = os.path.basename(args.smiles_file)
split = os.path.splitext(filename)
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
output_file = os.path.join(args.output_dir, "tabulate.csv")
if not os.path.exists(output_file):
    tab = pd.DataFrame()
else:
    tab = pd.read_csv(output_file)

res = pd.read_csv(args.smiles_file,header=None)
# res = res['output'].to_list()
try:
    res.columns = ['smiles','scaffold']
except:
    res.columns = ['smiles']
# print(data.head())
# train_data = pd.read_csv(args.reference_file, header=None)
train_data = pd.read_csv(args.reference_file)
try:
    train_data.columns = ['smiles']
except:
    train_data.columns = ['smiles','scaffold']
train_cannon_smiles = [canonic_smiles(s) for s in tqdm(train_data['smiles'])]
train_unique_smiles = list(set(train_cannon_smiles))
start_time = time.time()

canon_smiles = [canonic_smiles(s) for s in tqdm(res['smiles']) if canonic_smiles(s) is not None]
# write_smiles(canon_smiles,'./../model_sample/cleaned_gen_1000_aug_100.txt')
unique_smiles = np.unique(canon_smiles)
# write_smiles(unique_smiles,'./../model_sample/unique_aug_100_size_100.txt')
"""
export file for cfmid to predict spectrums and the style is like this:
id1 smiles
id2 smiles
"""
# len_unique_smiles = len(unique_smiles)
# cfmid_file = './../data/cfmid_smiles.txt'
#
# with open(cfmid_file,'w') as f:
#     for id,smi in zip(range(len_unique_smiles),unique_smiles):
#         _ = f.write(str(id)+ ' ' + smi +'\n')


# f1 = np.frompyfunc(lambda x:canonic_smiles(x),1,1)
# canon_smiles = f1(res['smiles'].values).astype(str)
# unique_smiles_np = np.unique(canon_smiles)
# unique_smiles = unique_smiles_np.tolist()
end_time = time.time()
print(f'time is {end_time - start_time}')
nove_smi = [i for i in unique_smiles if i not in train_unique_smiles]
write_smiles(nove_smi,'./../model_sample/novelty_1000.txt')
print(len(nove_smi))
novel_ratio = check_novelty(unique_smiles, set(train_unique_smiles))

"""
valid ratio: cannoic_smi/gen_smi
Novelty ratio:
Unique ratio:
"""

valid_ratio = len(canon_smiles) / (len(res))
print(len(canon_smiles))
unique_ratio = len(unique_smiles) / (len(canon_smiles))
print(len(unique_smiles))
novelty_ratio = novel_ratio


test_file = pd.read_csv(args.test_file, header=None)
test_file.columns = ['smiles']
test_cannon_smiles = [canonic_smiles(s) for s in tqdm(test_file['smiles'])]
test_unique_smiles = list(set(test_cannon_smiles))
# check_in_ratio = np.round(check_novelty(unique_smiles, set(test_unique_smiles))/100, 4)
check_in_ratio, duplicates, not_in = check_in(unique_smiles, test_unique_smiles)

tab = tab.append(pd.DataFrame({'file_name': [split[0]],
                               'valid_ratio': [round(valid_ratio, 4)],
                               'unique_ratio': [round(unique_ratio, 4)],
                               'novelty_ratio': [round(novelty_ratio, 4)],
                               'duplicates_ratio': [round(check_in_ratio, 4)],
                               'duplicates_num': [len(duplicates)],
                               'not_duplicates_num': [len(not_in)],
                               'test_num': [len(test_unique_smiles)],
                               }))
# tab.to_csv(output_file, index=False)

print('\ncheck in test is {:.4f}%'.format(check_in_ratio * 100))
print(f'\nduplicates is {duplicates},lenth of duplicates is {len(duplicates)}')
print(f'\nlenth of duplicates is {len(duplicates)}')
# print(f'\nnot in is {not_in},len of not in is {len(not_in)}')
print(f'\nnumber of not in is {len(not_in)}')
print(f'\nlen(test_unique_smiles) is {len(test_unique_smiles)}')
print(f'valid:{valid_ratio}\nunique:{unique_ratio}\nnovelty:{novel_ratio}')
