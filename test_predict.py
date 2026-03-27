import pickle
import pandas as pd

# Load model + columns
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Example input (raw human input)
input_data = {
    'company': 'Dell',
    'typename': 'Notebook',
    'inches': 15.6,
    'cpu': 'Intel',
    'ram': 8,
    'memory': 512,
    'gpu': 'Nvidia',
    'opsys': 'Windows',
    'weight': 2.0,
    'pixelcount': 2073600   # 1920x1080
}

# Convert manually (same preprocessing logic)
df = pd.DataFrame([input_data])

# Convert categorical
df = pd.get_dummies(df)

# Match training columns
df = df.reindex(columns=columns, fill_value=0)

# Predict
prediction = model.predict(df)

print("Predicted Price:", prediction[0])