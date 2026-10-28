import streamlit as st
import pickle
#from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem import AllChem
import numpy as np
from matplotlib import cm
#from ochem import mycalc
import pandas as pd
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings("ignore")


# Load the trained SVC model from pickle file
MODEL_FILE_alpha = 'best_model_alpha.pkl'  # Replace with your pickle file
MODEL_FILE_beta = 'best_model_beta.pkl' #best_model_FCFP8_XGB_RS25

# Function to load the model
def load_model(mymodel):
    with open(mymodel, 'rb') as f:
        model = pickle.load(f)
        model.probability=True
        #print(model)
    return model

# Function to generate molecular fingerprint
def generate_fingerprint(smiles,rad):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad,nBits=1024,useFeatures=True,useChirality = True)
    arr = np.zeros((1,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return mol, arr

def fpFunction_a(m,atomId=-1):
    fp = SimilarityMaps.GetMorganFingerprint(
        m,
        atomId=atomId,
        radius=2,
        nBits=1024,
        useFeatures=True,
        useChirality=True
    )
    return fp

def fpFunction_b(m, atomId=-1):
    fp = SimilarityMaps.GetMorganFingerprint(
        m,
        atomId=atomId,
        radius=4,
        nBits=1024,
        useFeatures=True,
        useChirality=True
    )
    return fp

def getProba(fp, predictionFunction):
    return predictionFunction((fp,))[0][1]

def plot_similarity_map(mol, model,fpFunction):
    d = Draw.MolDraw2DCairo(400, 400)
    SimilarityMaps.GetSimilarityMapForModel(mol,
                                            fpFunction,
                                            lambda x : float(getProba(x, model.predict_proba)),
                                            draw2d=d)
    d.FinishDrawing()
    return d






sdf1="LXR_training_Final_minimized.sdf"

sdf2="LXRB_rs25_tr_minimized.sdf"

def applicability(smiles,mysdf,rad):
    d_ECFP4 = {}
    m=Chem.MolFromSmiles(smiles)
    for mol in Chem.SDMolSupplier(mysdf):
        mg = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad,nBits=1024,useFeatures=True,useChirality = True)
        if m is not None:
           mg_ = AllChem.GetMorganFingerprintAsBitVect(m, rad, useFeatures=True)
           d_ECFP4.setdefault(Chem.MolToSmiles(m),[]).append(DataStructs.FingerprintSimilarity(mg, mg_))

    df_ECFP4 = pd.DataFrame.from_dict(d_ECFP4)
    print(df_ECFP4.max()[0])
    return df_ECFP4.max()[0]


# Streamlit application
def main():
    st.title("LXR PREDICTOR")

    st.subheader("This app predicts the binding potential of the chemical compounds towards Liver X receptor-alpha (LXRα) and Liver X receptor-beta (LXRβ) using trained fingerprint-based models")
    st.write("Input single or multiple (separated by comma) SMILES notation(s) of chemical compound(s) to predict activity.")

    # Input SMILES notation
    smi = st.text_input("Enter SMILES Notation:", "")
    slist=smi.split(',')
    for smi in slist:
        if smi:
           # Generate fingerprint
           mol, fingerprint_a = generate_fingerprint(smi,2)
           #mol, fingerprint_b = generate_fingerprint(smi,4)
           #act_trans,fig2,l1,l2=mycalc('fatimaBest.pickle',smiles)

           if mol is None:
              st.error("Invalid SMILES notation. Please try again.")
           else:
               # Load the model
               model_a = load_model(MODEL_FILE_alpha)
               model_b = load_model(MODEL_FILE_beta)

               # Predict activity for fingerprint based model
               prediction = model_a.predict([fingerprint_a])
               st.write("## Predicted activity, AD and similarity map (as per fingerprint-based model)")
               st.write("## LXRα activity")
               activity = "Active (EC50/IC50 < 1000 nM)" if prediction[0] == 1 else "Inactive (EC50/IC50 >= 1000 nM)"               
               st.write(f"Predicted Activity as per fingerprint based model: **{activity}**")

               #Check applicability domain
               value=applicability(smi,sdf1,2)
               if value>=0.3:
                  st.write("The compound falls within AD of of the model (as per fingerprint based model)")
               else:
                  st.write("The compound falls outside AD of of the model (as per fingerprint based model)")


               # Generate similarity map
               #d = Draw.MolDraw2DCairo(400, 400)


               #fig,_=SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, model.predict_proba), colorMap=cm.PiYG_r)
               #st.pyplot(fig)
               res=plot_similarity_map(mol, model_a,fpFunction_a)
               fig=res.GetDrawingText()
               st.image(fig)
               st.markdown("**Colour scheme:**")
               st.markdown('<span style="color:green">Molecular environments that increase predicted probability of class 1 (Active)</span>', unsafe_allow_html=True)
               st.markdown('<span style="color:red">Molecular environments that decrease predicted probability of class 0 (Inctive)</span>', unsafe_allow_html=True)
               
               #LXR-beta starts here 
               mol, fingerprint_b = generate_fingerprint(smi,4)
               prediction_b = model_b.predict([fingerprint_b])
               st.write("## LXRβ activity")
               activity = "Active (EC50/IC50 < 1000 nM)" if prediction_b[0] == 1 else "Inactive (EC50/IC50 >= 1000 nM)"
               #st.write("## Predicted activity, AD and similarity map (as per fingerprint-based model)")
               st.write(f"Predicted Activity as per fingerprint based model: **{activity}**")

               #Check applicability domain
               value=applicability(smi,sdf2,4)
               if value>=0.3:
                  st.write("The compound falls within AD of of the model (as per fingerprint based model)")
               else:
                  st.write("The compound falls outside AD of of the model (as per fingerprint based model)")


               # Generate similarity map
               #d = Draw.MolDraw2DCairo(400, 400)


               #fig,_=SimilarityMaps.GetSimilarityMapForModel(mol, fpFunction, lambda x: getProba(x, model.predict_proba), colorMap=cm.PiYG_r)
               #st.pyplot(fig)
               res_b=plot_similarity_map(mol,model_b,fpFunction_b)
               fig_b=res_b.GetDrawingText()
               st.image(fig_b)
               st.markdown("**Colour scheme:**")
               st.markdown('<span style="color:green">Molecular environments that increase predicted probability of class 1 (Active)</span>', unsafe_allow_html=True)
               st.markdown('<span style="color:red">Molecular environments that decrease predicted probability of class 0 (Inctive)</span>', unsafe_allow_html=True)




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
        Developed using Streamlit by Dr. Amit Kumar Halder, Professor, Dr. B. C. Roy College of Pharmacy and AHS, India | <a href="https://github.com/amitporto/LXR_PREDICTOR" target="_blank">Github link</a>
        This work is funded by the Department of Science and Technology and Biotechnology, Govt. of West Bengal, India Vide Memo. 2027 (Sanc.)/STBT-11012 (19)/ 6/2023-ST SEC, dated 24-01-2024.
    </div>
    """,
    unsafe_allow_html=True
)

if __name__ == "__main__":
    main()


