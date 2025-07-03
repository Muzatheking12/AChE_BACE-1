import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import pandas as pd
from streamlit_ketcher import st_ketcher
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from io import BytesIO

mainpath = os.path.dirname(__file__)
ACHEmodel = joblib.load(os.path.join(mainpath, r'ACHE/model.joblib'))
BACE1model = joblib.load(os.path.join(mainpath, r'BACE1/model.joblib'))
ACHEcol = os.path.join(mainpath, r'ACHE/col.csv')
BACE1col = os.path.join(mainpath, r'BACE1/col.csv')
smiledraw = os.path.join(mainpath, 'mol.png')
tanACHE = os.path.join(mainpath, r'ACHE/tanimoto_dist.png')
tanBACE1 = os.path.join(mainpath, r'BACE1/tanimoto_dist.png')
tsneACHE = os.path.join(mainpath, r'ACHE/tsne.png')
tsneBACE1 = os.path.join(mainpath, r'BACE1/tsne.png')
pcaAcHE = os.path.join(mainpath, r'ACHE/PCA.png')
pcaBACE1 = os.path.join(mainpath, r'BACE1/pca.png')
metACHE = os.path.join(mainpath, r'ACHE/Scatter_plot.png')
metBACE1 = os.path.join(mainpath, r'BACE1/Scatter_plot.png')
comACHE = os.path.join(mainpath, r'ACHE/Figure_1.png')
comBACE1 = os.path.join(mainpath, r'BACE1/Figure_1.png')
fpache = os.path.join(mainpath, r'ACHE/fingerprint.csv')
fpbace1 = os.path.join(mainpath, r'BACE1/fingerprint.csv')

st.title("🧪 Dual-Target AchE/BACE1 Inhibitor Prediction")
st.markdown("""
<div style='background-color:#f0f2f6; padding: 18px; border-radius: 10px;'>
<h2 style='color: #333;'>🔍 Overview</h2>

This <b>web application</b> integrates two <b>machine learning-based Regression models</b> to predict <b>dual-acting AChE/BACE1 inhibitors</b>. The Output is classified into <b>Inactives</b>, <b>Moderate</b>, <b>Actives</b> and <b>Strong Actives</b>. The Compound being Moderate, Active or Stong Active in each models within the Applicability Domain will result in DUAL_INHIBITION

<ul>
  <li>
    <b>Training Data:</b> 
    <ul>
      <li>AChE inhibitors: <b>4,528</b> compounds (CHEMBL220), pIC50 range: 1.3 - 11.22 </li>
      <li>BACE1 inhibitors: <b>5,278</b> compounds (CHEMBL4822), pIC50 range: 1.43 - 12.7 </li>
    </ul>
  </li>
  <li>
    <b>DUAL compounds</b> are predicted as inhibitors of both <b>AChE</b> and <b>BACE1</b> using this integrated model.
  </li>
  <li>
    <b>Applicability Domain (AD):</b> 
    <ul>
      <li><b>IN AD</b>: Compound is similar to the training set</li>
      <li><b>OUT AD</b>: Compound is not similar to the training set</li>
      <li>Adjust the AD threshold to filter out compounds with low similarity inhibitors.</li>
    </ul>
  </li>
  <li>
    <b>Model Validation:</b>
    <ul>
      <li>Validated with <b>10x10 K-fold cross-validation</b></li>
      <li><b>AChE</b> model:
      <li><b>R2<sub>cv</sub></b> = 0.710 ± 0.029</li>
      <li><b>RMSE<sub>cv</sub></b> = 0.786 ± 0.036</li>
      <li><b>BACE-1</b> model:   
      <li><b>R2<sub>cv</sub></b> = 0.707 ± 0.027</li>
      <li><b>RMSE<sub>cv</sub></b> = 0.694 ± 0.037</li>  
    </ul>
  </li>
  <li>
    <b>pIC50 Classification:</b>
    <ul>
      <li>pIC50 1 to < 5 = <b>Inactive</b></li>
      <li>pIC50 5 to < 6 = <b>Moderate</b></li>
      <li>pIC50 6 to 8 = <b>Active</b></li>   
      <li>pIC50 8 < = <b>Stong Active</b></li>
    </ul>
  </li>
  
</ul>
</div>
""", unsafe_allow_html=True)

st.write("## 📋 Instructions")
st.markdown("""
1. **🔬 Input SMILES**:  
   Enter a **SMILES string** and submit to predict **active/inactive** based on an **IC50 threshold of 1000nM**.
   
2. **🖌️ Draw Compounds**:  
   Use the **compound sketcher** and click **'Apply'** to copy the SMILES into the input field.
   
3. **⚙️ Adjust Thresholds**:  
   Modify the **AD threshold** to filter out compounds with low similarity.
   
4. **📂 Batch Input**:  
   Upload an **Excel file (.XLSX)** containing a column labeled **'SMILES'** for batch predictions.
""")



def tanimoto(fp1, fp2):
        intersection = np.sum(np.bitwise_and(fp1, fp2))
        union = np.sum(np.bitwise_or(fp1, fp2))

        if union == 0:
            return 0
        else:
            return intersection / union
def ext_tanimoto(threshold, fingerprinty, fingerprintx):
    external_molecule = np.array(fingerprinty)
    tanimoto_similarities = np.apply_along_axis(lambda row: tanimoto(row, external_molecule), 1, fingerprintx)
    sorted_indices = np.argsort(tanimoto_similarities)[::-1]
    top_k_indices = sorted_indices[1:6]
    mean_top_k_similarity = np.mean(tanimoto_similarities[top_k_indices])
    print("\nMean Tanimoto Similarity of Top 5 Neighbors:", mean_top_k_similarity)
    if mean_top_k_similarity >= threshold:
                                    AD = 'IN Applicability Domain'
    else:
                                    AD = 'OUT Applicability Domain'
    return AD

def compute_morgan_fingerprint(smiles, col, radius=2, n_bits=1024):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES string. Please check your input.")
            return None
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
         # Convert to numpy array and align with model features
        fp_arr = np.zeros((1, n_bits), dtype=int)
        AllChem.DataStructs.ConvertToNumpyArray(fingerprint, fp_arr[0])
        fp_df = pd.DataFrame(fp_arr, columns=[f"FP_{i+1}" for i in range(n_bits)])
        df = pd.read_csv(col)
        R = df.drop(columns=['Value'], axis=1)  # Assuming these are your feature columns
        common_columns = [col for col in fp_df.columns if col in R.columns]
        
        # Filter the fp_df to only include the common columns that exist in both
        X = fp_df[common_columns]

        # Convert DataFrame to numpy array for model input
        X = X.values # Ensure column order m
        y = R.values
        return X, y
    except Exception as e:
        st.error(f"Error computing fingerprint: {e}")
        return None

# Sidebar: SMILES input
sketched_smiles = st_ketcher()
smiles_input = st.sidebar.text_input("Enter a SMILES string:", sketched_smiles, placeholder="C1=CC=CC=C1")
button1 = st.sidebar.button("Submit")

# Sidebar: Upload Excel file
uploaded_file = st.sidebar.file_uploader("Upload an Excel file with SMILES", type=["xlsx"])

adjust_ACHE_threshold = st.sidebar.checkbox("Adjust AChE AD Threshold")
ACHE_threshold = st.sidebar.slider("AChE AD Threshold", 0.13, 1.0, 0.13, 0.01) if adjust_ACHE_threshold else 0.13

adjust_BACE1_threshold = st.sidebar.checkbox("Adjust BACE1 AD Threshold")
BACE1_threshold = st.sidebar.slider("BACE1 AD Threshold", 0.14, 1.0, 0.14, 0.01) if adjust_BACE1_threshold else 0.14

def classify_pic50_FP(df, column='pIC50'):
    def classify(value):
        try:
            value = float(value)
        except:
            return 'Compound'
        if value < 5:
            return 'Inactive'
        elif 5 <= value < 6:
            return 'Moderate'
        elif 6 <= value < 8:
            return 'Active'
        else:
            return 'Strong Active'
    df['Activity Class'] = df[column].apply(classify)
    return df



def classify_pic50(pred):
    if pred < 5:
        return "Inactive"
    elif 5 <= pred < 6:
        return "Moderate"
    elif 6 <= pred < 8:
        return "Active"
    else:
        return "Strong Active"


def GenerateCHEMspace(datapath, fingerprint):

    df = pd.read_csv(datapath)

    # Identify columns
    feature_columns = df.columns[:-1]
    label_column = df.columns[-1]

    if isinstance(fingerprint, np.ndarray):
     fingerprint = fingerprint.flatten().tolist()

    if len(fingerprint) != len(feature_columns):
      raise ValueError(f"Fingerprint length ({len(fingerprint)}) does not match feature columns ({len(feature_columns)})")

    df2 = pd.DataFrame([fingerprint + ['compound']], columns=list(feature_columns) + [label_column])

    combined_df = pd.concat([df, df2], ignore_index=True)

    # Apply classification
    inputdata = classify_pic50_FP(combined_df, column='Value')

    feature_cols = [col for col in inputdata.columns if col not in [label_column, 'Activity Class']]
    X = inputdata[feature_cols].values
    y = inputdata['Activity Class'].values

    # Encode classes
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = le.classes_

    # Color map
    color_map = {
        'Inactive': 'red',
        'Moderate': 'orange',
        'Active': 'blue',
        'Strong Active': 'green',
        'Compound': 'black'
    }
    colors = [color_map.get(label, 'gray') for label in y]

    # ----------- PCA Plot to BytesIO -------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig_pca, ax1 = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        idx = y_encoded == i
        ax1.scatter(X_pca[idx, 0], X_pca[idx, 1],
                    label=class_name, alpha=0.7, edgecolor='k',
                    color=color_map[class_name])
    ax1.scatter(X_pca[-1, 0], X_pca[-1, 1], color='black', edgecolor='yellow', s=120, marker='*', label='Input Compound')
    ax1.set_title("PCA Plot")
    ax1.set_xlabel("PCA-1")
    ax1.set_ylabel("PCA-2")
    ax1.legend()
    ax1.grid(True)

    pca_buffer = BytesIO()
    fig_pca.savefig(pca_buffer, format='png', bbox_inches='tight')
    pca_buffer.seek(0)
    plt.close(fig_pca)

    # ----------- t-SNE Plot to BytesIO -------------
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    fig_tsne, ax2 = plt.subplots(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        idx = y_encoded == i
        ax2.scatter(X_tsne[idx, 0], X_tsne[idx, 1],
                    label=class_name, alpha=0.7, edgecolor='k',
                    color=color_map[class_name])
    ax2.scatter(X_tsne[-1, 0], X_tsne[-1, 1], color='black', edgecolor='yellow', s=120, marker='*', label='Input Compound')
    ax2.set_title("t-SNE Plot")
    ax2.set_xlabel("t-SNE-1")
    ax2.set_ylabel("t-SNE-2")
    ax2.legend()
    ax2.grid(True)

    tsne_buffer = BytesIO()
    fig_tsne.savefig(tsne_buffer, format='png', bbox_inches='tight')
    tsne_buffer.seek(0)
    plt.close(fig_tsne)

    return pca_buffer, tsne_buffer






tab1, tab2, tab3 = st.tabs(["Prediction Results", "AChE", "BACE1"])
if button1:
        with tab1:
            m = Chem.MolFromSmiles(smiles_input, sanitize=False)
            AllChem.Compute2DCoords(m)

            # Save the 2D structure as an image file
        
            img = Draw.MolToImage(m)
            img.save(smiledraw)
            st.image(smiledraw)
            fingerprint1, y1 = compute_morgan_fingerprint(smiles_input, ACHEcol)
            AD_ACHE = ext_tanimoto(ACHE_threshold, fingerprint1, y1)
            prediction1 = ACHEmodel.predict(fingerprint1)[0]
            fingerprint2, y2 = compute_morgan_fingerprint(smiles_input, BACE1col)
            AD_BACE1 = ext_tanimoto(BACE1_threshold, fingerprint2, y2)
            prediction2 = BACE1model.predict(fingerprint2)[0]
            
           
            st.subheader("Predictions")
            class1 = classify_pic50(prediction1)
            class2 = classify_pic50(prediction2)

            if class1 in ["Moderate", "Active", "Strong Active"] and class2 in ["Moderate", "Active", "Strong Active"] and AD_ACHE == 'IN Applicability Domain' and AD_BACE1 == 'IN Applicability Domain':
                st.success("The Compound is predicted to be **DUAL INHIBITOR** AcHE/BACE1.")
                st.write(f"AChE : **{class1}** (pIC₅₀: {prediction1:.2f}) (**{AD_ACHE}**)")
                st.write(f"BACE1 : **{class2}** (pIC₅₀: {prediction2:.2f}) (**{AD_BACE1}**)")
               
               
            else:
                st.write(f"AChE : **{class1}** (pIC₅₀: {prediction1:.2f}) (**{AD_ACHE}**)")
                st.write(f"BACE1 : **{class2}** (pIC₅₀: {prediction2:.2f}) (**{AD_BACE1}**)")
                
               
              
                
          
                #st.write(f"AChE  : **{'Active' if prediction1 == 1 else 'Inactive'}** (**{AD_ACHE}**)")
                #st.write(f"Similarity with Predicted DUAL Inhibitors: **{AD_DUAL}**")
                #st.write(f"**SHAP Contribution: Red = Positive Contribution, Blue = Negative Contribution**")
                #images = generate_shap_image(shap.TreeExplainer(ACHEmodel), ACHEmodel, smiles_input, ACHEcol)
                #st.pyplot(images)
                #st.write(f"BACE1 : **{'Active' if prediction2 == 1 else 'Inactive'}** (**{AD_BACE1}**)")
                #st.write(f"Similarity with Predicted DUAL Inhibitors: **{AD_DUAL}**")
                #st.write(f"**SHAP Contribution: Red = Positive Contribution, Blue = Negative Contribution**")
                #images = generate_shap_image(shap.Explainer(BACE1model), BACE1model, smiles_input, BACE1col)
                #st.pyplot(images)

smiles_list = []

# Add SMILES from uploaded file
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if "SMILES" in df.columns:
            smiles_list.extend(df["SMILES"].dropna().tolist())
        else:
            st.sidebar.error("The uploaded file must contain a 'SMILES' column.")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Create a table for SMILES, predictions, and AD

if smiles_list:
  with tab1:
    st.subheader("Predictions for Uploaded SMILES")
    results = []  # To store results for the table
    for smiles in smiles_list:
        try:
            fingerprint1, y1 = compute_morgan_fingerprint(smiles, ACHEcol)
            AD_ACHE = ext_tanimoto(ACHE_threshold, fingerprint1, y1) if fingerprint1 is not None else "N/A"
            prediction1 = ACHEmodel.predict(fingerprint1)[0] if fingerprint1 is not None else None
            class1 = classify_pic50(prediction1) if prediction1 is not None else "N/A"

            fingerprint2, y2 = compute_morgan_fingerprint(smiles, BACE1col)
            AD_BACE1 = ext_tanimoto(BACE1_threshold, fingerprint2, y2) if fingerprint2 is not None else "N/A"
            prediction2 = BACE1model.predict(fingerprint2)[0] if fingerprint2 is not None else None
            class2 = classify_pic50(prediction2) if prediction2 is not None else "N/A"

            

            is_dual = (
                class1 in ["Moderate", "Active", "Strong Active"] and
                class2 in ["Moderate", "Active", "Strong Active"] and
                AD_ACHE == "IN Applicability Domain" and
                AD_BACE1 == "IN Applicability Domain"
            )

            results.append({
                "SMILES": smiles,
                "DUAL INHIBITION": "YES" if is_dual else "NO",
                "AChE pIC₅₀": round(prediction1, 2) if prediction1 is not None else "N/A",
                "AChE Class": class1,
                "AChE AD": AD_ACHE,
                "BACE1 pIC₅₀": round(prediction2, 2) if prediction2 is not None else "N/A",
                "BACE1 Class": class2,
                "BACE1 AD": AD_BACE1,
            
            })
        except Exception as e:
            st.error(f"Error processing SMILES {smiles}: {e}")

    results_df = pd.DataFrame(results)
    st.subheader("Predictions Table")
    st.dataframe(results_df)
    


with tab2:
    st.header("AChE Model")
    st.write("### Model Overview")
    st.write("### Pairwise Tanimoto Similarity Distribution")
    st.image(tanACHE)
    st.write("### t-SNE Visualization")
    st.image(tsneACHE)
    st.write("### PCA Visualization")
    st.image(pcaAcHE)
    st.write("### Scatter Plot of Predicted vs Actual")
    st.write("**Model: Random Forest Regressor**")
    st.write("**FP: Morgan**")
    st.image(metACHE)
    st.write("### Comparative Analysis")
    st.image(comACHE)

with tab3:
       st.header("BACE1 Model")
       st.write("### Model Overview")
       st.write("### Pairwise Tanimoto Similarity Distribution")
       st.image(tanBACE1)
       st.write("### t-SNE Visualization")
       st.image(tsneBACE1)
       st.write("### PCA Visualization")
       st.image(pcaBACE1)
       st.write("### Scatter Plot of Predicted vs Actual")
       st.write("**Model: NuSVR**")
       st.write("**FP: Morgan**")
       st.image(metBACE1)
       st.write("### Comparative Analysis")
       st.image(comBACE1)

