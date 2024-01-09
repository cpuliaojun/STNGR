import pandas as pd

data = pd.read_csv('./../data/test_scaffolds.csv')
data['SMILES'].to_csv('./../data/moss_scaffolds_test.txt',header=None,index=False)