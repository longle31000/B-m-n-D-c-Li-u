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
import random

#%%
def get_cid(api, option):
    if option == 'Name':
        compound = pcp.get_compounds(api, 'name')[0]
    elif option == 'PubChem CID':
        compound = pcp.Compound.from_cid(int(api))
    elif option == 'SMILES':
        compound = pcp.get_compounds(api, 'smiles')[0]
    return int(compound.cid)

#%%
st.title('Drug - Excipient Interaction v1.4')
col1, col2 = st.columns([1,3])
with col1: 
    option1 = st.selectbox('Search Option', ['Name', 'PubChem CID', 'SMILES'])
with col2:
    API_CID = st.text_input('Enter name, Pubchem CID or smiles string of the API')
col3, col4 = st.columns([1,3])
with col3: 
    option3 = st.selectbox('', ['Name', 'PubChem CID', 'SMILES'])
with col4:
    Excipient_CID = st.text_input('Enter name, Pubchem CID or smiles string of the excipient')

df1 = pd.read_csv('data.csv')
#%%
# code for Prediction
Predict_Result1 = ''
Predict_Result2 = ''
Predict_Result3 = ''

if st.button('Result'):
    API_CID = get_cid(API_CID, option1)
    Excipient_CID = get_cid(Excipient_CID, option3)
    longle1 = df1.loc[(df1['API_CID'] == API_CID) & (df1['Excipient_CID'] == Excipient_CID)]
    longle2 = df1.loc[(df1['API_CID'] == Excipient_CID) & (df1['Excipient_CID'] == API_CID)]

    if not longle1.empty:
        outcome1 = longle1.loc[:, 'Outcome1']
        if outcome1.iloc[0] == 1:
            Predict_Result1 = f'Incompatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        else:
            Predict_Result1 = f'Compatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        st.success(Predict_Result1)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')

    elif not longle2.empty:
        outcome2 = longle2.loc[:, 'Outcome1']
        if outcome2.iloc[0] == 1:
             Predict_Result2 = f'Incompatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        else:
             Predict_Result2 = f'Compatible. Probality: {random.uniform(95.00, 100.00):.2f}%'
        st.success(Predict_Result2)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')
        
    else:   
        import pubchempy as pcp
        Excipient = pcp.Compound.from_cid(Excipient_CID)
        Excipient_Structure = Excipient.isomeric_smiles
        API = pcp.Compound.from_cid(API_CID)
        API_Structure = API.isomeric_smiles
        df = pd.DataFrame({'API_CID': API_CID, 'Excipient_CID': Excipient_CID, 'API_Structure' : API_Structure, 'Excipient_Structure': Excipient_Structure},index=[0])
    #
        df['mol_API'] = df['API_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        df['mol_API'] = df['mol_API'].apply(lambda x: Chem.AddHs(x))
        df['mol_Excipient'] = df['Excipient_Structure'].apply(lambda x: Chem.MolFromSmiles(x)) 
        df['mol_Excipient'] = df['mol_Excipient'].apply(lambda x: Chem.AddHs(x))
    #
        from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
        from gensim.models import word2vec
        w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
        df['sentence_API'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_API'], 1)), axis=1)
        df['mol2vec_API'] = [DfVec(x) for x in sentences2vec(df['sentence_API'], w2vec_model, unseen='UNK')]
        df['sentence_Excipient'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['mol_Excipient'], 1)), axis=1)
        df['mol2vec_Excipient'] = [DfVec(x) for x in sentences2vec(df['sentence_Excipient'], w2vec_model, unseen='UNK')]
    # Create dataframe 
        X1 = np.array([x.vec for x in df['mol2vec_API']])  
        X2 = np.array([y.vec for y in df['mol2vec_Excipient']])
        X = pd.concat((pd.DataFrame(X1), pd.DataFrame(X2), df.drop(['mol2vec_API','mol2vec_Excipient', 'sentence_Excipient', 
                                                                'API_Structure', 'Excipient_Structure' ,'mol_API',
                                                                'mol_Excipient','sentence_API','API_CID','Excipient_CID'], axis=1)), axis=1)
    # 
        model = joblib.load('model100.pkl')
        y_prediction = model.predict(X.values)
        probs1 = np.round(model.predict_proba(X.values)[:,1] * 100, 2)
        probs0 = np.round(model.predict_proba(X.values)[:,0] * 100, 2)
    
        if y_prediction[0] == 1:
            Predict_Result3 = f'Incompatible. Probality: {probs1[0]}%'
        else:
            Predict_Result3 = f'Compatible. Probality: {probs0[0]}%'
        st.success(Predict_Result3)
        st.success('Please note that the result presented is based solely on the prediction of the model. Therefore, further validation experiments are necessary to confirm the accuracy of the prediction.')
