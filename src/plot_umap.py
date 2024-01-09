import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import umap
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs,Descriptors,Draw

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

def plot_umap():
    cata = {'Aminoindanes': 0,
            'Arylalkylamines': 1,
            'Arylcyclohexylamines': 2,
            'Benzodiazepines': 3,
            'Cannabinoids': 4,
            'Cathinones': 5,
            'Indolalkylamines': 6,
            'Opioids': 7,
            'Phenethylamines': 8,
            'Piperazine derivates': 9,
            'Piperidines & pyrrolidines': 10,
            'Plants & extracts': 11,
            'Precursors': 12,
            'Unknown': 13,
            'picked': 14}
    cata_reve = {0:'Aminoindanes',
            1:'Arylalkylamines',
            2:'Arylcyclohexylamines',
            3:'Benzodiazepines',
            4:'Cannabinoids',
            5:'Cathinones',
            6:'Indolalkylamines',
            7:'Opioids',
            8:'Phenethylamines',
            9:'Piperazine derivates',
            10:'Piperidines & pyrrolidines',
            11:'Plants & extracts',
            12:'Precursors',
            13:'Unknown',
            14:'picked'}

    umap_data_path = '../data/nps/clean_NPS_class_train.csv'
    umap_data = pd.read_csv(umap_data_path)
    umap_data['fp'] = umap_data['clean_SMILES'].apply(calFP)
    data = umap_data['fp']
    data = pd.DataFrame(data, columns=['fp'])
    x = np.array([x.fp for x in data['fp']])
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(x)
    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    labels = np.array(umap_data['DrugClass'])
    labels = np.array([cata[string] for string in labels])
    embedding_df['Label'] = labels

    plt.figure(figsize=(10, 6))
    # colors = ['red', 'blue', 'green', 'silver', 'orange', 'pink', 'brown', 'gray', 'cyan',
    #           'magenta', 'yellow', 'lime', 'black', 'white', 'purple']
    colors = ['red', 'red', 'red', 'red', 'green', 'red', 'red', 'red', 'red',
              'red', 'red', 'red', 'red', 'red', 'yellow']
    for label, color in zip(range(15), colors):
        subset = embedding_df[embedding_df['Label'] == label]
        plt.scatter(subset['UMAP1'], subset['UMAP2'], c=color, label=f'Class {cata_reve[label]}', alpha=0.5)
    plt.title('UMAP Visualization with Class Labels')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.show()


def plot_can():
    cata = {'Cannabinoids': 0,
            'picked': 1}
    cata_reve = {0:'Cannabinoids',
                 1:'picked'}
    data_path = '../data/nps/clean_NPS_class_train.csv'
    df = pd.read_csv(data_path)
    # print(df)
    grouped = df.groupby('DrugClass')
    for name, group in grouped:
        if name == 'Cannabinoids':
            can_group = group
        if name == 'picked':
            picked_group = group
    df = pd.concat([can_group, picked_group], ignore_index=True)
    df['fp'] = df['clean_SMILES'].apply(calFP)
    data = df['fp']
    data = pd.DataFrame(data, columns=['fp'])
    x = np.array([x.fp for x in data['fp']])
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(x)
    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    labels = np.array(df['DrugClass'])
    smiles = np.array(df['clean_SMILES'])
    labels = np.array([cata[string] for string in labels])
    embedding_df['Label'] = labels
    embedding_df['SMILES'] = smiles
    embedding_df.to_csv('../data/nps/can_picked1.csv', index=False)

    plt.figure(figsize=(10, 6))
    colors = ['green', 'red']
    for label, color in zip(range(2), colors):
        subset = embedding_df[embedding_df['Label'] == label]
        plt.scatter(subset['UMAP1'], subset['UMAP2'], c=color, label=f'Class {cata_reve[label]}', alpha=0.5)
    plt.title('UMAP Visualization with Class Labels')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.show()

def plot_can_csv():
    cata = {'Cannabinoids': 0,
            'picked': 1,
            'JWH-018': 2}
    cata_reve = {0:'Cannabinoids',
                 1:'picked',
                 2:'JWH-018'}
    embedding_df = pd.read_csv('../data/nps/can_picked1.csv')
    labels = np.array(embedding_df['Label'])
    # labels = np.array([cata[string] for string in labels])

    plt.figure(figsize=(10, 6))
    colors = ['green', 'red', 'blue']
    for label, color in zip(range(3), colors):
        subset = embedding_df[embedding_df['Label'] == label]
        plt.scatter(subset['UMAP1'], subset['UMAP2'], c=color, label=f'{cata_reve[label]}', alpha=0.5)
    plt.title('Cannabinoids chemical space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.savefig('UMAP.png')
    plt.show()


if __name__ == '__main__':
    plot_can_csv()