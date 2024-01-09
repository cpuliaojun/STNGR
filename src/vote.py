import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from outmodel import sascorer
from rdkit import Chem
from rdkit.Chem import AllChem,Descriptors

# sns.set_palette(sns.color_palette("Blues_r",10))
# sns.set()
import numpy as np
import math

def read_smiles(smiles_file):
    """
    Read a list of SMILES from a line-delimited file.
    """
    smiles = []
    with open(smiles_file, 'r') as f:
        smiles.extend([line.strip() for line in f.readlines() \
                       if line.strip()])
    return smiles

# train_test = pd.read_csv('./../data/nps/annotated_nps_aid.csv')
# test = train_test[train_test['group']==4]
# test.to_csv('./../data/test.csv',index=False)
# test['smiles_no_salt_canon'].to_csv('./../data/test.txt',header=None,index=False)

# test = read_smiles('./../data/nps/NpsTest.txt')

test = read_smiles('./../data/nps/2023_test_nps.txt')
# res = pd.read_csv('./../data/ensemble/aug_ensemble_results_0.99.csv')
res = pd.read_csv('./../data/ensemble/aug_ensemble_results_0.99_sa.csv')
# res.columns = ['smiles','Ensemble (n prediction(s) in most active class)']
sorted_res = pd.read_csv('./../cal_outcome/frequency_results.csv')
sorted_res.columns = ['SMILES','num']



# res['sa'] = res['SMILES'].apply(lambda x:sascorer.calculateScore(Chem.MolFromSmiles(x)))
# res.to_csv('./../data/ensemble/aug_ensemble_results_0.99_sa.csv',index=False)
# res['qed'] = res['SMILES'].apply(lambda x:Descriptors.qed(Chem.MolFromSmiles(x)))
top_100 = res[res['Ensemble (n prediction(s) in most active class)'] >= 95]
top_100 = top_100[top_100['sa']<=3]

merge_df = pd.merge(top_100,sorted_res,how='inner',on='SMILES')
merge_df['vote'] = merge_df['Ensemble (n prediction(s) in most active class)']/100
merge_df['freq'] = (merge_df['num'] - merge_df['num'].min())/(merge_df['num'].max()-merge_df['num'].min())
# merge_df['score'] = merge_df['vote']-merge_df['sa']/10
merge_df['score'] = merge_df['vote']-merge_df['sa']/10 +merge_df['freq']
# top_100 = merge_df.sort_values(by='num',ascending=False).iloc[:3500,:]
top_100 = merge_df.sort_values(by='score',ascending=False).iloc[:500,:].reset_index()

pd.set_option('display.max_columns',None)
top_100_li = top_100['SMILES'].to_list()
out = [i for i in top_100_li if i in test]
# print(out)
print(len(out))
print(out)


# def plot():
# res = pd.read_csv('./../data/ensemble/aug_ensemble_results_0.99.csv')
res = res[res['sa']<=2]
# res = res[res['qed']>=0.7]
count = res['Ensemble (n prediction(s) in most active class)'].value_counts()

df = pd.DataFrame({'vote':count.index,'counts':count})
print(df)
dff = df.iloc[:9,:]
# df['counts'] = df['counts'].apply(lambda x:math.log(x,10))
# count.columns = ['vote','counts']
fig,ax = plt.subplots()
# ax.set_ylim([-0.1, 1.05])
# ax.set_xlim([80, 100])
ax.bar(dff['vote'],dff['counts'])
ax.set_xlabel('Number of votes')
ax.set_ylabel('Number of Molecules')
# sns.barplot(data=dff,x='vote',y='counts')
# plt.savefig('./../data/count-vote.png',dpi=600)
plt.show()





