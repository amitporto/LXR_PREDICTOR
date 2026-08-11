import streamlit as st
import pickle
#from rdkit import Chem
#from rdkit import Chem, DataStructs
#from rdkit.Chem import Draw
#from rdkit.Chem.Draw import SimilarityMaps
#from rdkit.Chem import AllChem
import numpy as np
from matplotlib import cm
#from ochem import mycalc
import pandas as pd
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

st.write("Application started")

import rdkit

st.write("RDKit imported")
st.write(rdkit.__version__)

from rdkit import Chem

st.write("Chem imported")


# Load the trained SVC model from pickle file
MODEL_FILE = 'best_model.pkl'  # Replace with your pickle file


# Function to load the model
def load_model():
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
        #model.probability=True
        print(model)
    return model

# Function to generate molecular fingerprint
def generate_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2,nBits=1024,useFeatures=True,useChirality = True)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return mol, arr

def fpFunction(m, atomId=-1):
    fp = SimilarityMaps.GetMorganFingerprint(m, atomId=atomId, radius=2, nBits=1024,useChirality = True)
    return fp

def getProba(fp, predictionFunction):
    return predictionFunction((fp,))[0][1]


d_ECFP4 = {}




sdf1="training_fp_minimized.sdf"

#sdf2="training_tc_minimized.sdf"

def applicability(smiles,sdf):
    m=Chem.MolFromSmiles(smiles)
    for mol in Chem.SDMolSupplier("LXR_training_Final_minimized.sdf"):
        mg = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2,nBits=1024,useFeatures=True,useChirality = True)
        if m is not None:
           mg_ = AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True)
           d_ECFP4.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))

    df_ECFP4 = pd.DataFrame.from_dict(d_ECFP4)
    print(df_ECFP4.max()[0])
    return df_ECFP4.max()[0]

def plot_similarity_map(mol, model):
    d = Draw.MolDraw2DCairo(400, 400)
    SimilarityMaps.GetSimilarityMapForModel(mol,
                                            fpFunction,
                                            lambda x : getProba(x, model.predict_proba),
                                            draw2d=d)
    d.FinishDrawing()
    return d

# Streamlit application
def main():
    st.title("LXR PREDICTOR")

    st.subheader("This app predicts the binding potential of the chemical compounds towards Liver X receptors (LXR) as per fingerprint-based model")
    st.write("Input a SMILES notation of a chemical compound to predict its activity.")

    # Input SMILES notation
    smi = st.text_input("Enter SMILES Notation:", "")
    slist=smi.split(',')
    for smi in slist:
        if smi:
           # Generate fingerprint
           mol, fingerprint = generate_fingerprint(smi)
           #act_trans,fig2,l1,l2=mycalc('fatimaBest.pickle',smiles)

           if mol is None:
              st.error("Invalid SMILES notation. Please try again.")
           else:
               # Load the model
               model = load_model()

               # Predict activity for fingerprint based model
               prediction = model.predict([fingerprint])
               activity = "Active (EC50/IC50 < 1000 nM)" if prediction[0] == 1 else "Inactive (EC50/IC50 >= 1000 nM)"
               st.write("## Predicted activity, AD and similarity map (as per fingerprint-based model)")
               st.write(f"Predicted Activity as per fingerprint based model: **{activity}**")

               #Check applicability domain
               value=applicability(smi,sdf1)
               if value>=0.3:
                  st.write("The compound falls within AD of of the model (as per fingerprint based model)")
               else:
                  st.write("The compound falls outside AD of of the model (as per fingerprint based model)")


               # Generate similarity map
               d = Draw.MolDraw2DCairo(400, 400)


               #fig,_=SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, model.predict_proba), colorMap=cm.PiYG_r)
               #st.pyplot(fig)
               res=plot_similarity_map(mol, model)
               fig=res.GetDrawingText()
               st.image(fig)
               st.markdown("**Colour scheme:**")
               st.markdown('<span style="color:green">The fragments of the molecule that increase the binding potential of the compound</span>', unsafe_allow_html=True)
               st.markdown('<span style="color:red">The fragments of the molecule that decrease the binding potential of the compound</span>', unsafe_allow_html=True)




# Add a footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #f1f1f1;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    </style>
    <div class="footer">
        Developed using Streamlit by Dr. Amit Kumar Halder, Professor, Dr. B. C. Roy College of Pharmacy and AHS, India | <a href="https://bcrcp.ac.in/" target="_blank">About Us</a>
        This work is funded by the Department of Science and Technology and Biotechnology, Govt. of West Bengal, India Vide Memo. 2027 (Sanc.)/STBT-11012 (19)/ 6/2023-ST SEC, dated 24-01-2024.
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()


