# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:10:52 2023

@author: WINDOWS
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import joblib
import pubchempy as pcp
import numpy as np                
import pandas as pd               
import matplotlib.pyplot as plt   
import seaborn as sns
from rdkit.Chem import Descriptors
from rdkit import RDLogger  
RDLogger.DisableLog('rdApp.*')  
from rdkit import Chem 

def get_cid(api):
    # Check if input is a PubChem ID
    if api.isdigit():
        compound = pcp.Compound.from_cid(int(api))
        return int(compound.cid)
    # Check if input is a SMILES string
    elif pcp.get_compounds(api, 'smiles'):
        compound = pcp.get_compounds(api, 'smiles')[0]
        return int(compound.cid)

st.title('Drug - Excipient Interaction')
col1, col2 = st.columns(2)
with col1: 
    API_CID = st.text_input('Number of API_CID')
with col2:
    Excipient_CID = st.text_input('Number of Excipient_CID')

uploaded_file = st.file_uploader("Choose your file", type='csv')
if uploaded_file is not None:
    df1 = pd.read_csv(uploaded_file)
else:
    df1 = pd.read_csv("hoa2.csv")
    
# code for Prediction
Predict_Result1 = ''
Predict_Result2 = ''
Predict_Result3 = ''
if st.button('Result'):
    API_CID = get_cid(API_CID)
    Excipient_CID = get_cid(Excipient_CID)
    longle1 = df1.loc[(df1['API_CID'] == API_CID) & (df1['Excipient_CID'] == Excipient_CID)]
    longle2 = df1.loc[(df1['API_CID'] == Excipient_CID) & (df1['Excipient_CID'] == API_CID)]

    if not longle1.empty:
        outcome1 = longle1.loc[:, 'Outcome1']
        if outcome1.iloc[0] == 1:
            Predict_Result1 = 'Incompatible'
        else:
            Predict_Result1 = 'Compatible'
        st.success(Predict_Result1)

    elif not longle2.empty:
        outcome2 = longle2.loc[:, 'Outcome1']
        if outcome2.iloc[0] == 1:
             Predict_Result2 = 'Incompatible'
        else:
             Predict_Result2 = 'Compatible'
        st.success(Predict_Result2)       
        
    else:   
        import pubchempy as pcp
        Excipient = pcp.Compound.from_cid(Excipient_CID)
        Excipient_Structure = Excipient.isomeric_smiles
        API = pcp.Compound.from_cid(API_CID)
        API_Structure = API.isomeric_smiles
        df = pd.DataFrame({'API_CID': API_CID, 'Excipient_CID': Excipient_CID, 'API_Structure' : API_Structure, 'Excipient_Structure': Excipient_Structure},index=[0])
    ####
        df['mol_API'] = df['API_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        df['mol_API'] = df['mol_API'].apply(lambda x: Chem.AddHs(x))
    ####
        from rdkit.Chem import Descriptors
        df['MolWt_API'] = df['mol_API'].apply(lambda x: Descriptors.MolWt(x))
        df['MolLogP_API'] = df['mol_API'].apply(lambda x: Descriptors.MolLogP(x))
        df['MolMR_API'] = df['mol_API'].apply(lambda x: Descriptors.MolMR(x))
        df['HeavyAtomCount_API'] = df['mol_API'].apply(lambda x: Descriptors.HeavyAtomCount(x))
        df['NumHAcceptors_API'] = df['mol_API'].apply(lambda x: Descriptors.NumHAcceptors(x))
        df['NumHDonors_API'] = df['mol_API'].apply(lambda x: Descriptors.NumHDonors(x))
        df['NumHeteroatoms_API'] = df['mol_API'].apply(lambda x: Descriptors.NumHeteroatoms(x))
        df['NumRotatableBonds_API'] = df['mol_API'].apply(lambda x: Descriptors.NumRotatableBonds(x))
        df['NumValenceElectrons_API'] = df['mol_API'].apply(lambda x: Descriptors.NumValenceElectrons(x))      
        df['NumAromaticRings_API'] = df['mol_API'].apply(lambda x: Descriptors.NumAromaticRings(x))
        df['NumSaturatedRings_API'] = df['mol_API'].apply(lambda x: Descriptors.NumSaturatedRings(x))
        df['NumAliphaticRings_API'] = df['mol_API'].apply(lambda x: Descriptors.NumAliphaticRings(x))
        df['RingCount_API'] = df['mol_API'].apply(lambda x: Descriptors.RingCount(x))
        df['TPSA_API'] = df['mol_API'].apply(lambda x: Descriptors.TPSA(x))
        df['LabuteASA_API'] = df['mol_API'].apply(lambda x: Descriptors.LabuteASA(x))
        df['BalabanJ_API'] = df['mol_API'].apply(lambda x: Descriptors.BalabanJ(x))
        df['BertzCT_API'] = df['mol_API'].apply(lambda x: Descriptors.BertzCT(x))
    ####
        df['mol_Excipient'] = df['Excipient_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        df['mol_Excipient'] = df['mol_Excipient'].apply(lambda x: Chem.AddHs(x))
    ####
        df['MolWt_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.MolWt(x))
        df['MolLogP_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.MolLogP(x))
        df['MolMR_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.MolMR(x))
        df['HeavyAtomCount_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.HeavyAtomCount(x))
        df['NumHAcceptors_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumHAcceptors(x))
        df['NumHDonors_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumHDonors(x))
        df['NumHeteroatoms_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumHeteroatoms(x))
        df['NumRotatableBonds_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumRotatableBonds(x))
        df['NumValenceElectrons_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumValenceElectrons(x))      
        df['NumAromaticRings_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumAromaticRings(x))
        df['NumSaturatedRings_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumSaturatedRings(x))
        df['NumAliphaticRings_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.NumAliphaticRings(x))
        df['RingCount_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.RingCount(x))
        df['TPSA_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.TPSA(x))
        df['LabuteASA_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.LabuteASA(x))
        df['BalabanJ_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.BalabanJ(x))
        df['BertzCT_Excipient'] = df['mol_Excipient'].apply(lambda x: Descriptors.BertzCT(x))
        import rarfile
        from gensim.models import Word2Vec

   
    #### 
        from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
        from gensim.models import word2vec
        with rarfile.RarFile('model_300dim.rar') as rf:
            rf.extractall()  # giải nén tất cả các file trong rar
            with open('model_300dim.pkl', 'rb') as f:
                w2vec_model = pickle.load(f)
        df['sentence_API'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_API'], 1)), axis=1)
        df['mol2vec_API'] = [DfVec(x) for x in sentences2vec(df['sentence_API'], w2vec_model, unseen='UNK')]
        df['sentence_Excipient'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_Excipient'], 1)), axis=1)
        df['mol2vec_Excipient'] = [DfVec(x) for x in sentences2vec(df['sentence_Excipient'], w2vec_model, unseen='UNK')]
    #### Tạo dataframe 
        X1 = np.array([x.vec for x in df['mol2vec_API']])  
        X2 = np.array([y.vec for y in df['mol2vec_Excipient']])
        X = pd.concat((pd.DataFrame(X1), pd.DataFrame(X2), df.drop(['mol2vec_API','mol2vec_Excipient', 'sentence_Excipient', 
                                                                'API_Structure', 'Excipient_Structure' ,'mol_API',
                                                                'mol_Excipient','sentence_API', 'MolWt_API','MolLogP_API','MolMR_API','HeavyAtomCount_API',
                                                            'NumHAcceptors_API', 'NumHDonors_API',
'NumHeteroatoms_API', 'NumRotatableBonds_API', 'NumValenceElectrons_API','NumAromaticRings_API','NumSaturatedRings_API',
'NumAliphaticRings_API','RingCount_API','TPSA_API','LabuteASA_API','BalabanJ_API','BertzCT_API',
'MolWt_Excipient','MolLogP_Excipient','MolMR_Excipient','HeavyAtomCount_Excipient','NumHAcceptors_Excipient',
'NumHDonors_Excipient','NumHeteroatoms_Excipient', 'NumRotatableBonds_Excipient', 'NumValenceElectrons_Excipient',
'NumAromaticRings_Excipient','NumSaturatedRings_Excipient','NumAliphaticRings_Excipient','RingCount_Excipient',
'TPSA_Excipient','LabuteASA_Excipient','BalabanJ_Excipient','BertzCT_Excipient', 'API_CID', 'Excipient_CID' ], axis=1)), axis=1)
        import joblib
        model = joblib.load('model100.pkl')
        diab_prediction = model.predict(X)
    
        if diab_prediction[0] == 1:
            Predict_Result3 = 'Incompatible'
        else:
            Predict_Result3 = 'Compatible'
        st.success(Predict_Result3)




    
    



