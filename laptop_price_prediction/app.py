import streamlit as st
import pickle
import pandas as pd

# Load
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("💻 Laptop Price Predictor")

# Inputs
company = st.selectbox("Company", ["Dell", "HP", "Apple", "Asus", "Acer"])
typename = st.selectbox("Type", ["Notebook", "Gaming", "Ultrabook"])
inches = st.slider("Screen Size", 10.0, 18.0, 15.6)
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
memory = st.selectbox("Storage (GB)", [128, 256, 512, 1024])
cpu = st.selectbox("CPU Brand", ["Intel", "AMD"])
gpu = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD"])
opsys = st.selectbox("OS", ["Windows", "Mac", "Linux"])
weight = st.slider("Weight (kg)", 1.0, 4.0, 2.0)

# Approx resolution
pixelcount = 1920 * 1080

if st.button("Predict Price"):

    input_data = {
        'inches': inches,
        'ram': ram,
        'memory': memory,
        'weight': weight,
        'pixelcount': pixelcount,
        f'company_{company}': 1,
        f'typename_{typename}': 1,
        f'cpu_{cpu}': 1,
        f'gpu_{gpu}': 1,
        f'opsys_{opsys}': 1
    }

    df = pd.DataFrame([input_data])
    df = df.reindex(columns=columns, fill_value=0)

    price = model.predict(df)[0]
    Estimated_price=0-price
    rounded_price=round(Estimated_price,2)

    st.success(f"💰 Estimated Price: ₹{rounded_price}")