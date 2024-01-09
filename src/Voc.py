# -*- encoding: utf-8 -*-
"""
@file         :    Voc.py
@description  :    vocabulary python file
@date         :    2022/8/11 17:23
@author       :    silentpotato  
"""
from utils import read_smiles
from itertools import chain
import re
import pandas as pd
import numpy as np
from rdkit import Chem


class Vocabulary():

    def __init__(self, smiles=None, scaf=None, smiles_file=None, vocab_file=None,smi_max_len=None,
                 scaf_max_len=None):
        self.special_tokens = ['EOS', 'GO']
        self.scaf = scaf
        self.max_len = smi_max_len
        self.scaf_max_len = scaf_max_len
        pattern = "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(pattern)
        if vocab_file:
            self.characters = read_smiles(vocab_file)
        else:
            # read SMILES
            if smiles:
                self.smiles = smiles
            elif smiles_file:
                self.smiles = read_smiles(smiles_file)
            else:
                raise ValueError("must provide SMILES list or file to" + \
                                 " instantiate Vocabulary")
            # tokenize all SMILES in the input and add all tokens to vocabulary
            all_chars = [self.tokenize(sm) for sm in self.smiles] + [self.tokenize(sm) for sm in self.scaf]
            # all_chars = [self.tokenize(sm) for sm in self.smiles]
            self.characters = sorted(list(set(chain(*all_chars))))
        # add padding token
        if not '<PAD>' in self.characters:
            # ... unless reading a padded vocabulary from file
            self.characters.append('<PAD>')

        # create dictionaries
        # self.pos_chars = ['<PAD>','<SOS>','<EOS>']
        self.pos_chars = ['<PAD>']
        self.dictionary = {key: idx for idx, key in
                           enumerate([item for item in self.characters if item not in self.pos_chars], start=1)}
        self.dictionary['<PAD>'] = 0
        # self.dictionary['<SOS>'] = 1
        # self.dictionary['<EOS>'] = 2
        self.reverse_dictionary = {value: key for key, value in \
                                   self.dictionary.items()}


    def tokenize(self,smiles):

        tokens_without_padding = self.regex.findall(smiles.strip())
        tokens = tokens_without_padding + ['<PAD>'] * (self.max_len - len(tokens_without_padding))
        return tokens
    def scaf_tokenize(self, scaf):
        tokens_without_padding = self.regex.findall(scaf.strip())
        tokens = tokens_without_padding + ['<PAD>'] * (self.scaf_max_len - len(tokens_without_padding))
        return tokens
    # REGEXPS = {
    #     "brackets": re.compile(r"(\[[^\]]*\])"),
    #     "2_ring_nums": re.compile(r"(%\d{2})"),
    #     "brcl": re.compile(r"(Br|Cl)")
    # }
    # REGEXP_ORDER = ["brackets", "2_ring_nums", "brcl"]

    # def tokenize(self, smiles):
    #     """
    #     Convert a SMILES string into a sequence of tokens.
    #     """
    #
    #     def split_by(smiles, regexps):
    #         if not regexps:
    #             return list(smiles)
    #         regexp = self.REGEXPS[regexps[0]]
    #         splitted = regexp.split(smiles)
    #         tokens = []
    #         for i, split in enumerate(splitted):
    #             if i % 2 == 0:
    #                 tokens += split_by(split, regexps[1:])
    #             else:
    #                 tokens.append(split)
    #         return tokens
    #
    #     tokens_without_padding = split_by(smiles, self.REGEXP_ORDER)
    #     tokens = tokens_without_padding + ['<PAD>'] * (self.max_len - len(tokens_without_padding))
    #     return tokens

    # def tokenize(self, smiles):
    #     try:
    #         tokens = self.regex.findall(smiles.strip())
    #     except AttributeError as e:
    #         print(smiles)
    #
    #     tokens = tokens + ['<PAD>'] * (self.max_len - len(tokens))
    #     return tokens

    # def scaf_tokenize(self, scaf):
    #     def split_by(smiles, regexps):
    #         if not regexps:
    #             return list(smiles)
    #         regexp = self.REGEXPS[regexps[0]]
    #         splitted = regexp.split(smiles)
    #         tokens = []
    #         for i, split in enumerate(splitted):
    #             if i % 2 == 0:
    #                 tokens += split_by(split, regexps[1:])
    #             else:
    #                 tokens.append(split)
    #         return tokens
    #
    #     tokens_with_padding = split_by(scaf, self.REGEXP_ORDER)
    #     tokens = tokens_with_padding + ['<PAD>'] * (self.scaf_max_len - len(tokens_with_padding))
    #     return tokens


        # tokens = self.regex.findall(scaf.strip())
        # tokens = tokens + ['<PAD>'] * (self.scaf_max_len - len(tokens))
        # return tokens

    def encode(self, char_list):
        """
        Encode a series of tokens into a (numeric) tensor.
        """
        # vec = torch.zeros(len(tokens)+2)
        # for idx, token in enumerate(tokens):
        #     vec[idx] = self.dictionary[token]
        # return vec.long()
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.dictionary[char]
        return smiles_matrix

    def decode(self, sequence):
        """
        Decode a series of tokens back to a SMILES.
        """
        sequence = sequence.tolist()
        # chars = []
        # for i in sequence:
        #     if i == self.dictionary['<EOS>']:
        #         break
        #     if i != self.dictionary['<SOS>']:
        #         chars.append(self.reverse_dictionary[i])
        # smiles = "".join(chars)
        completion = ''.join([self.reverse_dictionary[int(i)] for i in sequence])
        smi = completion.replace('<PAD>', '')
        # for gen_mol in y:
        #     completion = ''.join([reverse_dic[int(i)] for i in gen_mol])
        #     completion = completion.replace('<PAD>', '')
        #     gen_smiles.append(completion)


        # smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smi

    def char_write(self, output_file):
        """
        Write the list of tokens in a vocabulary to a line-delimited file.
        """
        with open(output_file, 'w') as f:
            for char in self.characters:
                f.write(char + '\n')

    # def scaf_char_write(self,output_file):
    #     with open(output_file, 'w') as f:
    #         for char in self.scaf_characters:
    #             f.write(char + '\n')

    def __len__(self):
        return len(self.characters)

    def __str__(self):
        return "vocabulary containing " + str(len(self)) + " characters: " + \
               format(self.characters)
class Experience(object):
    """Class for prioritized experience replay that remembers the highest scored sequences
       seen and samples from them with probabilities relative to their scores."""

    def __init__(self, voc, max_size=100):
        self.memory = []
        self.max_size = max_size
        self.voc = voc

    def add_experience(self, experience):
        """Experience should be a list of (smiles, score, prior likelihood) tuples"""
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, smiles = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in smiles:
                    idxs.append(i)
                    smiles.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key=lambda x: x[1], reverse=True)
            self.memory = self.memory[:self.max_size]
            print("\nBest score in memory: {:.2f}".format(self.memory[0][1]))

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            sample = np.random.choice(len(self), size=n, replace=False, p=scores / np.sum(scores))
            sample = [self.memory[i] for i in sample]
            smiles = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        return encoded, np.array(scores), np.array(prior_likelihood)

    def initiate_from_file(self, fname, scoring_function, Prior):
        """Adds experience from a file with SMILES
           Needs a scoring function and an RNN to score the sequences.
           Using this feature means that the learning can be very biased
           and is typically advised against."""
        with open(fname, 'r') as f:
            smiles = []
            for line in f:
                smile = line.split()[0]
                if Chem.MolFromSmiles(smile):
                    smiles.append(smile)
        scores = scoring_function(smiles)
        tokenized = [self.voc.tokenize(smile) for smile in smiles]
        encoded = [Variable(self.voc.encode(tokenized_i)) for tokenized_i in tokenized]
        encoded = MolData.collate_fn(encoded)
        prior_likelihood, _ = Prior.likelihood(encoded.long())
        prior_likelihood = prior_likelihood.data.cpu().numpy()
        new_experience = zip(smiles, scores, prior_likelihood)
        self.add_experience(new_experience)

    def print_memory(self, path):
        """Prints the memory."""
        print("\n" + "*" * 80 + "\n")
        print("         Best recorded SMILES: \n")
        print("Score     Prior log P     SMILES\n")
        with open(path, 'w') as f:
            f.write("SMILES Score PriorLogP\n")
            for i, exp in enumerate(self.memory[:100]):
                if i < 50:
                    print("{:4.2f}   {:6.2f}        {}".format(exp[1], exp[2], exp[0]))
                    f.write("{} {:4.2f} {:6.2f}\n".format(*exp))
        print("\n" + "*" * 80 + "\n")

    def __len__(self):
        return len(self.memory)






if __name__ == "__main__":
    # train_val_smiles = read_smiles('./../data/smi.txt')
    train_val_file = pd.read_csv('./../data/moses_raw/dataset_v1.csv')
    # new = train_val_file.dropna()
    # new.to_csv('./../data/moses_train_scaffold.csv',header=None,index=False)
    # train_val_file.columns = ['smiles', 'scaffold']
    vocabulary = Vocabulary(train_val_file['SMILES'].tolist(),train_val_file['SMILES'].tolist(), smi_max_len=100, scaf_max_len=128)
    # vocabulary = Vocabulary(vocab_file='./../data/voc.txt',smi_max_len=128,scaf_max_len=100)
    # vocabulary.char_write('./../data/moses_voc.txt')
    # vocabulary.scaf_char_write('./../data/scaf_voc.txt')
    print(vocabulary.dictionary)
    print(len(vocabulary.characters))

