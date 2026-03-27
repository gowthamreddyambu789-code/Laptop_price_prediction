import pandas as pd
import numpy as np
import re

# Load data
data = pd.read_csv("laptop_data.csv")

# -------------------------------
# 1. CLEAN RAM
# -------------------------------
data['Ram'] = data['Ram'].str.replace('GB', '').astype(int)

# -------------------------------
# 2. CLEAN WEIGHT
# -------------------------------
data['Weight'] = data['Weight'].str.replace('kg', '').astype(float)

# -------------------------------
# 3. SCREEN RESOLUTION → PixelCount
# -------------------------------
def extract_resolution(x):
    match = re.search(r'(\d+)x(\d+)', x)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        return width * height
    return 0

data['PixelCount'] = data['ScreenResolution'].apply(extract_resolution)
data.drop('ScreenResolution', axis=1, inplace=True)

# -------------------------------
# 4. CPU (simplify to brand)
# -------------------------------
data['Cpu'] = data['Cpu'].apply(lambda x: x.split()[0])

# -------------------------------
# 5. MEMORY (convert to GB)
# -------------------------------
def convert_memory(x):
    x = x.replace('GB', '').replace('TB', '000')
    x = x.split('+')
    total = 0
    for item in x:
        item = item.strip()
        if 'SSD' in item or 'HDD' in item:
            num = re.findall(r'\d+', item)
            if num:
                total += int(num[0])
    return total

data['Memory'] = data['Memory'].apply(convert_memory)

# -------------------------------
# 6. GPU (brand only)
# -------------------------------
data['Gpu'] = data['Gpu'].apply(lambda x: x.split()[0])

# -------------------------------
# 7. DROP LESS USEFUL COLUMN
# -------------------------------
# (optional, but improves model)
# data.drop('OpSys', axis=1, inplace=True)

# -------------------------------
# 8. HANDLE CATEGORICAL DATA
# -------------------------------
data = pd.get_dummies(data, drop_first=True)

# -------------------------------
# 9. SPLIT DATA
# -------------------------------
X = data.drop('Price', axis=1)
y = data['Price']

# -------------------------------
# 10. TRAIN MODEL
# -------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 11. SAVE MODEL + COLUMNS
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model trained successfully!")