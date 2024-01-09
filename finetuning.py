# -*- encoding: utf-8 -*-
"""
@file         :    train.py
@description  :
@date         :    2022/8/12 15:36
@author       :    silentpotato
"""
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
import pandas as pd
from utils import read_smiles
from utils import get_max_lenth
from torch.cuda.amp import GradScaler
import time
from datasets import SmilesDataset
from Voc import Vocabulary
import pickle
from model import GPT,sample
from tqdm import tqdm

def get_data_and_targets(batch):
    data = batch[:,:-1]
    targets = batch[:,1:]
    data = data.to(device)
    targets = targets.to(device)
    return data,targets

def train():
    model.train()
    total_loss = 0.
    total_acc = 0.
    # start_time = time.time()
    it = tqdm(enumerate(loader,start=1), total=len(loader))
    for batch_idx, (smi_batch, scaf_batch) in it:
        data, targets = get_data_and_targets(smi_batch)
        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(True):
                output,_ = model(data,scaf_batch.to(data.device))
                pred = output.argmax(dim=-1)
                correct = torch.eq(pred,targets).sum().float().item()
                acc = correct / data.shape[0] / data.shape[1]
                output = output.view(-1, ntokens)
                loss = criterion(output, targets.reshape(-1))
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()
        total_acc += acc
        # loss = loss.mean()
        it.set_description(f"epoch {epoch} iter {batch_idx} train loss {loss.item():.5f} acc {acc:.5f} lr {args.lr:e} ")
    #     if batch_idx % 200 == 0 and batch_idx > 0:
    #         cur_loss = total_loss / 200
    #         elapsed = time.time() - start_time
    #         print('|epoch: {} |train_loss: {:.4f} |time: {:.4f}'.format(epoch,cur_loss,elapsed))
    #         total_loss = 0
    #         start_time = time.time()
    cur_loss = total_loss / batch_idx
    cur_acc = acc / batch_idx
    return cur_loss,cur_acc

def evaluate():
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    total_acc = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (smi_batch, scaf_batch) in enumerate(vloader,start=1):
            data, targets = get_data_and_targets(smi_batch)
            output, _ = model(data, scaf_batch.to(data.device))
            pred = output.argmax(dim=-1)
            correct = torch.eq(pred, targets).sum().float().item()
            acc = correct / data.shape[0] / data.shape[1]
            output = output.view(-1, ntokens)
            loss = criterion(output, targets.reshape(-1))
            total_loss += loss.item()
            total_acc += acc
        cur_acc = total_acc / batch_idx
        cur_loss = total_loss / batch_idx
        elapsed = time.time() - start_time
        print('|epoch: {} |val_loss : {:.4f}| val_acc : {:.4f} |time: {:.4f}'.format(epoch, cur_loss,acc, elapsed))
        print('='*47)
    return cur_loss,cur_acc

def check_point(save_path,model_name):
    if not os.path.isdir('{}'.format(save_path)):
        os.mkdir('{}'.format(save_path))
    torch.save(model.state_dict(), save_path+model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_scaffold_file', type=str,
                        default='./../data/nps_train_general_smiles_scaffold.csv')
    parser.add_argument('--vocab_file', type=str, default='./../data/voc.txt')
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=12)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=512)

    # parser.add_argument('--save', type=str, default='./../model_file/finetuning_model.pt', help='path to save the final model')
    parser.add_argument('--clip', type=float, default=1, help='gradient clipping')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')

    parser.add_argument('--check_point_path', type=str, default='./../checkpoint/')
    parser.add_argument('--smiles_max_len', type=int, default=90)
    parser.add_argument('--scaf_max_len', type=int, default=70)

    parser.add_argument('--pretrain_model',default='./../checkpoint/encoder_moses.pt')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    train_val_file = pd.read_csv(args.smiles_scaffold_file, header=None)
    train_val_file.columns = ['smiles', 'scaffold']
    # train_data = train_val_file.sample(frac=0.9, random_state=3407)
    # val_data = train_val_file[~train_val_file.index.isin(train_data.index)]
    train_data = train_val_file
    val_data = train_val_file
    train_val_smiles = train_val_file['smiles'].tolist()
    train_val_scaf = train_val_file['scaffold'].tolist()
    smiles_max_len = get_max_lenth(train_val_smiles)
    scaf_max_len = get_max_lenth(train_val_scaf)
    print(f'smiles_len in dataset:{smiles_max_len}\nscaf_len in dataset:{scaf_max_len}')

    vocabulary = Vocabulary(vocab_file=args.vocab_file, smi_max_len=args.smiles_max_len,
                            scaf_max_len=args.scaf_max_len)
    print(f'in Voc,max lenth of smiles is {args.smiles_max_len}\nmax lenth of scaf is {args.scaf_max_len}')

    train_dataset = SmilesDataset(train_data['smiles'].tolist(), vocabulary=vocabulary,
                                  scaffolds=train_data['scaffold'].tolist())
    v_dataset = SmilesDataset(val_data['smiles'].tolist(), vocabulary=vocabulary,
                              scaffolds=val_data['scaffold'].tolist())

    loader = DataLoader(train_dataset, shuffle=True, pin_memory=True, batch_size=args.batch_size)
    vloader = DataLoader(v_dataset, shuffle=True, pin_memory=True, batch_size=args.val_batch_size)
    smi_dic = vocabulary.dictionary
    reverse_dic = vocabulary.reverse_dictionary

    criterion = nn.NLLLoss()

    ntokens = len(vocabulary)

    pre_model = GPT(
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
    pre_model.load_state_dict(torch.load(args.pretrain_model))
    #冻结模型
    # for param in pre_model.parameters():
    #     param.requires_grad = False
    # 微调
    # for param in pre_model.blocks[-1].mlp.parameters():
    #     param.requires_grad=True
    # for param in pre_model.projection.parameters():
    #     param.requires_grad = True
    # in_features, out_features = pre_model.projection.in_features, pre_model.projection.out_features
    # pre_model.projection = nn.Linear(in_features, out_features, bias=False)
    model = pre_model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.projection.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scaler = GradScaler()

    best_loss = float('inf')
    plot_df = pd.DataFrame({'epoch': [], 'train_loss': [],'train_acc':[] ,'val_loss': [],'val_acc':[]})
    for epoch in range(1, args.epochs + 1):
        train_loss,train_acc = train()
        val_loss,val_acc = evaluate()
        temp_df = pd.DataFrame([[epoch, train_loss,train_acc, val_loss,val_acc]], columns=plot_df.columns)
        plot_df = plot_df.append(temp_df)
        if val_loss < best_loss:
            best_loss = val_loss
            check_point(args.check_point_path,'moses_finetuning_general_model.pt')
        # print(f'train file saving epoch {epoch} model...')
    # with open('./../data/loss/finetuning_loss.pkl','wb') as f:
    #     pickle.dump(plot_df,f)
