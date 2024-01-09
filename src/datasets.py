# -*- encoding: utf-8 -*-
"""
@file         :    data_sets.py   
@description  :
@date         :    2022/8/12 16:29
@author       :    silentpotato  
"""

import torch
from torch.utils.data import Dataset
from utils import read_smiles
from Voc import Vocabulary


class SmilesDataset(Dataset):
    """
    A dataset of chemical structures, provided in SMILES format.
    """

    def __init__(self, smiles=None, smiles_file=None, vocab_file=None,
                 vocabulary=None, scaffolds=None):
        """
        Can be initiated from either a list of SMILES, or a line-delimited
        file.

        Args:
            smiles (list): the complete set of SMILES that constitute the
              training dataset
            smiles_file (string): line-delimited file containing the complete
              set of SMILES that constitute the training dataset
            vocab_file (string): line-delimited file containing all tokens to
              be used in the vocabulary
            training_split (numeric): proportion of the dataset to withhold for
              validation loss calculation
            vocabulary(Vocabulary)
        """
        if smiles:
            self.smiles = smiles
        elif smiles_file:
            self.smiles = read_smiles(smiles_file)
        else:
            raise ValueError("must provide SMILES list or file to" + \
                             " instantiate SmilesDataset")
        # create vocabulary
        if vocab_file:
            self.vocabulary = Vocabulary(vocab_file=vocab_file)
        else:
            # self.vocabulary = Vocabulary(smiles=self.smiles,max_len=max_len)
            self.vocabulary = vocabulary
        self.scaffolds = scaffolds

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        # return Variable(self.vocabulary.encode(
        #         self.vocabulary.tokenize(self.training[idx])))
        smi = self.smiles[idx]
        scaf = self.scaffolds[idx]
        smi_tokens = self.vocabulary.tokenize(smi)
        scaf_tokens = self.vocabulary.scaf_tokenize(scaf)
        smi_tensor = torch.tensor([self.vocabulary.dictionary[token] for token in smi_tokens], dtype=torch.long)
        scaf_tensor = torch.tensor([self.vocabulary.dictionary[token] for token in scaf_tokens], dtype=torch.long)
        # return torch.tensor([self.vocabulary.dictionary[token] for token in smi_tokens],dtype=torch.long)
        return smi_tensor, scaf_tensor

    def __str__(self):
        return "dataset containing " + str(len(self)) + \
               " SMILES with a vocabulary of " + str(len(self.vocabulary)) + \
               " characters"

    def get_vocabulary(self):
        return self.vocabulary
