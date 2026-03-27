import streamlit as st
import pickle
import pandas as pd
import os

# -------------------------------
# 🔍 DEBUG (VERY IMPORTANT)
# -------------------------------
st.write("Current directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# -------------------------------
# 📂 LOAD MODEL SAFELY
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
columns_path = os.path.join(BASE_DIR, "columns.pkl")

model = pickle.load(open(model_path, "rb"))
columns = pickle.load(open(columns_path, "rb"))

# -------------------------------
# 🎯 UI
# -------------------------------
st.title("💻 Laptop Price Predictor")

company = st.selectbox("Company", ["Dell", "HP", "Apple", "Asus", "Acer"])
typename = st.selectbox("Type", ["Notebook", "Gaming", "Ultrabook"])
inches = st.slider("Screen Size (inches)", 10.0, 18.0, 15.6)
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32])
memory = st.selectbox("Storage (GB)", [128, 256, 512, 1024])
cpu = st.selectbox("CPU Brand", ["Intel", "AMD"])
gpu = st.selectbox("GPU Brand", ["Intel", "Nvidia", "AMD"])
opsys = st.selectbox("Operating System", ["Windows", "Mac", "Linux"])
weight = st.slider("Weight (kg)", 1.0, 4.0, 2.0)

# Approx resolution (fixed for simplicity)
pixelcount = 1920 * 1080

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
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

    prediction = model.predict(df)[0]

    # ✅ Prevent negative price
    prediction = max(0, prediction)

    st.success(f"💰 Estimated Price: ₹{prediction:.2f}")

   
