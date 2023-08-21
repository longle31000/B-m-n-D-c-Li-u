#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from rdkit import Chem
import os

#%%
df= pd.read_csv('F:/Phuong/Project/API Excipient interaction/New word2vec/Raw data.csv')
print(df.head())
df['mol_API'] = df['API_Structure'].apply(lambda x: Chem.MolFromSmiles(x))
df['mol_Excipient'] = df['Excipient_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
df['mol_Excipient'] = df['mol_Excipient'].apply(lambda x: Chem.AddHs(x))
df['mol_API'] = df['mol_API'].apply(lambda x: Chem.AddHs(x))
#%%
os.chdir(r'F:\Phuong\Project\API Excipient interaction\New word2vec')
open('model_300dim.pkl', 'rb')
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
df['sentence_API'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_API'], 1)), axis=1)
df['mol2vec_API'] = [DfVec(x) for x in sentences2vec(df['sentence_API'], w2vec_model, unseen='UNK')]
df['sentence_Excipient'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_Excipient'], 1)), axis=1)
df['mol2vec_Excipient'] = [DfVec(x) for x in sentences2vec(df['sentence_Excipient'], w2vec_model, unseen='UNK')]
#
X1 = np.array([x.vec for x in df['mol2vec_API']])  
X2 = np.array([y.vec for y in df['mol2vec_Excipient']])
X = pd.concat((pd.DataFrame(X1), pd.DataFrame(X2), df.drop(['mol2vec_API','mol2vec_Excipient', 'sentence_Excipient', 
                                                                'API_Structure', 'Excipient_Structure' ,'mol_API',
                                                                'mol_Excipient','sentence_API','API_CID', 'Excipient_CID' ], axis=1)), axis=1)
print(X)
print(X.shape)

X.to_csv('F:/Phuong/Project/API Excipient interaction/New word2vec/data.csv', sep=',', encoding='utf-8', index=False)

#%%
