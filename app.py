import streamlit as st
import pandas as pd
import joblib
from difflib import get_close_matches
# ----------------------
# Load Models and Data
# ----------------------
@st.cache_data
def load_similarity():
    df= pd.read_csv("cleaned_data.csv",index_col=False)
    sim_df=pd.read_csv('product_similarity.csv', index_col=0)
    return df,sim_df

@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    return scaler, kmeans

df,similarity_df = load_similarity()
scaler, kmeans = load_models()

# ---------------------
# Helper Functions
# ---------------------

# 🔁 Mapping: StockCode ↔ Product Name
product_map = df[["StockCode", "Description"]].drop_duplicates().dropna()
product_map["StockCode"] = product_map["StockCode"].astype(str)  # 🔁 ensure all codes are str
code_to_name = dict(zip(product_map["StockCode"], product_map["Description"]))
name_to_code = {v: k for k, v in code_to_name.items()}

def get_closest_product_name(input_name):
    all_names = list(name_to_code.keys())
    match = get_close_matches(input_name, all_names, n=1, cutoff=0.6)
    return match[0] if match else None

def recommend_products(product_name):
    match_name = get_closest_product_name(product_name)
    if not match_name:
        return ["❌ No close match found for the product name."]
    
    product_code = name_to_code.get(match_name)
    if not product_code or product_code not in similarity_df.columns:
        return ["❌ Product not found in similarity matrix."]
    
    top5_codes = [str(code) for code in similarity_df[product_code].sort_values(ascending=False)[1:6].index.tolist()]

    top5_names = [code_to_name.get(code, f"Unknown Product ({code})") for code in top5_codes]
    
    return [f"{i+1}. {name}" for i, name in enumerate(top5_names)]

def assign_segment(r, f, m, rfm_ref):
    if r <= rfm_ref['Recency'].quantile(0.25) and f >= rfm_ref['Frequency'].quantile(0.75) and m >= rfm_ref['Monetary'].quantile(0.75):
        return 'High-Value'
    elif f >= rfm_ref['Frequency'].median() and m >= rfm_ref['Monetary'].median():
        return 'Regular'
    elif f < rfm_ref['Frequency'].quantile(0.25) and m < rfm_ref['Monetary'].quantile(0.25) and r > rfm_ref['Recency'].quantile(0.75):
        return 'At-Risk'
    else:
        return 'Occasional'

# Load full RFM for threshold-based labeling
rfm_ref = pd.read_csv('rfm_clusters.csv', index_col=0)

# ---------------------
# Streamlit App Layout
# ---------------------
st.set_page_config(page_title="🛍 Shopper Spectrum", layout="centered")
st.title("Shopper Spectrum")
tab1, tab2 = st.tabs(["🔁 Product Recommendation", "🎯 Customer Segmentation"])

# ---------------------
# Tab 1: Product Recommendation
# ---------------------
with tab1:
    st.header("🔁 Recommend Similar Products")

    product_names = sorted(name_to_code.keys())
    selected_product = st.selectbox("Enter Product Name",product_names)

    if st.button("Recommend Products"):
        recommendations = recommend_products(selected_product)
        st.success("Top 5 Recommended Products:")
        for rec in recommendations:
            st.write(rec)

# ---------------------
# Tab 2: Customer Segmentation
# ---------------------
with tab2:
    st.header("🎯 Customer Segmentation (RFM)")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0)

    if st.button("Predict Cluster"):
        input_df = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
        scaled_input = scaler.transform(input_df)
        cluster_label = kmeans.predict(scaled_input)[0]

        segment = assign_segment(recency, frequency, monetary, rfm_ref)
        st.write(f"🧠 Predicted Cluster: `{cluster_label}`")
        st.write(f"📌 **Segment Label**: `{segment}`")
