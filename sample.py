import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import math
from tqdm import tqdm
from model import GPT,sample
from Voc import Vocabulary
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

parser = argparse.ArgumentParser()
parser.add_argument('--scaffold_file', type=str,
            default=f'./../data/2154_scaf.txt')

parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--n_layers', type=int, default=12)
parser.add_argument('--dropout', type=float, default=0.3)

parser.add_argument('--vocab_file', type=str, default='./../data/voc.txt')
parser.add_argument('--sample_path', type=str, default='./../model_sample/')
parser.add_argument('--smiles_max_len', type=int, default=90)
parser.add_argument('--scaf_max_len', type=int, default=80)


parser.add_argument('-scaf_file',default='')
parser.add_argument('-sample_size',default=1000)
parser.add_argument('-sample_batch_size',default=1000)
parser.add_argument('--sample_temperature', type=float, default=1.0)

parser.add_argument('--checkpoint',default='./../checkpoint/nps/nps_200_bm_scaffold.pt')
args = parser.parse_args()
vocabulary = Vocabulary(vocab_file=args.vocab_file, smi_max_len=args.smiles_max_len,
                        scaf_max_len=args.scaf_max_len)


# model_name = args.checkpoint.split('/')[-1].split('.')[0]
smi_dic = vocabulary.dictionary
reverse_dic = vocabulary.reverse_dictionary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def sample_smiles_from_scaffold(scaffold_file):
    context = 'C'
    ntokens = len(vocabulary)
    model = GPT(
        ntokens,
        args.smiles_max_len,
        args.scaf_max_len,
        args.emb_size,
        args.n_head,
        args.n_layers,
        args.dropout,
        args.dropout,
        args.dropout,
        lstm=False,
        encoder=True
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    gen_iter = math.ceil(args.sample_size / args.sample_batch_size)
    # scaf_li = read_smiles(scaffold_file)
    scaf_li = pd.read_csv('./../data/nps/smiles_scaffold_1_2143.csv',header=None)
    scaf_li.columns = ['smiles', 'scaffold']
    # scaf_li = scaf_li.dropna(how='any',axis=0)
    scaf_li = scaf_li['scaffold'].to_list()
    # scaf_li = [i for i in scaf_li if i is not None]
    # scaf_li = ['CC(CC1CCC2CCCCC2C(C)C1C)C1CCC(CC2CCCCC2)C1']
    # scaf_li = ['CCC(C)C(C1CCCCC1)C1CCC(CCC2CCCCC2)CC1']
    for scaf in tqdm(scaf_li):
        num = 0
        for i in range(gen_iter):
            x = torch.tensor([smi_dic[context]], dtype=torch.long)[None, ...].repeat(args.sample_batch_size,
                                                                 1).to('cuda')
            sca = vocabulary.scaf_tokenize(scaf)
            scaffold = torch.tensor([smi_dic[s] for s in sca], dtype=torch.long)[None, ...].repeat(args.sample_batch_size,
                                                                                                        1).to('cuda')
            y = sample(model,vocabulary,x,steps=args.smiles_max_len,scaf=scaffold,temperature=args.sample_temperature)
            gen_smiles = []
            for gen_mol in y:
                completion = ''.join([reverse_dic[int(i)] for i in gen_mol])
                completion = completion.replace('<PAD>', '')
                gen_smiles.append(completion)
                # gen_smiles,n = filter(gen_smiles)
                # if gen_smiles == []:
                #     continue
            with open(f'{args.sample_path}nps_aug_withoutVC{args.sample_size}_{args.sample_temperature}.txt','a') as f:
                for item in gen_smiles:
                    _ = f.write(item+'\n')


def sample_smiles_from_scaf(scaf):
    context = 'C'
    ntokens = len(vocabulary)
    model = GPT(
        ntokens,
        args.smiles_max_len,
        args.scaf_max_len,
        args.emb_size,
        args.n_head,
        args.n_layers,
        args.dropout,
        args.dropout,
        args.dropout,
        lstm=False,
        encoder=True
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    gen_iter = math.ceil(args.sample_size / args.sample_batch_size)

    scaf_li = scaf
    for scaf in tqdm(scaf_li):
        num = 0
        for i in range(gen_iter):
            x = torch.tensor([smi_dic[context]], dtype=torch.long)[None, ...].repeat(1, 1).to('cuda')
            sca = vocabulary.scaf_tokenize(scaf)
            scaffold = torch.tensor([smi_dic[s] for s in sca], dtype=torch.long)[None, ...].repeat(1, 1).to('cuda')
            y = sample(model,vocabulary,x,steps=args.smiles_max_len,scaf=scaffold,temperature=args.sample_temperature)
            gen_smiles = []
            for gen_mol in y:
                completion = ''.join([reverse_dic[int(i)] for i in gen_mol])
                completion = completion.replace('<PAD>', '')
                gen_smiles.append(completion)

                return gen_smiles

def plot_mol(smi):
    smiles = smi[0]
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    return img



if __name__ == '__main__':
    # sample_smiles_from_scaffold(args.scaffold_file)
    scaf_li = ['c1ccc2c(c1)OCO2']
    gen_smiles = sample_smiles_from_scaf(scaf_li)
    print(gen_smiles)
    img = plot_mol(gen_smiles)
    img.save('mol.jpg')
    # ['C(C)(NC(C(=O)c1ccc2c(c1)OCO2)CCCC)C']