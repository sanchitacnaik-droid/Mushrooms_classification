import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Mushroom Classification Dashboard", page_icon="🍄")

st.title("🍄 Mushroom Classification Dashboard")

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("mushrooms.csv")

st.subheader("Dataset")
st.dataframe(df)

# -------------------------------
# Dataset Shape
# -------------------------------
st.subheader("Dataset Shape")

st.write(f"Rows : {df.shape[0]}")
st.write(f"Columns : {df.shape[1]}")
st.write(f"Datapoints : {df.size}")

# -------------------------------
# Checkboxes
# -------------------------------
if st.checkbox("Show First Five Rows"):
    st.write(df.head())

if st.checkbox("Column Names"):
    st.write(df.columns)

if st.checkbox("Missing Values"):
    st.write(df.isnull().sum())

if st.checkbox("Duplicate Rows"):
    st.write(df.duplicated().sum())

# -------------------------------
# Categorical Columns
# -------------------------------
if st.checkbox("Categorical Columns"):
    st.write(df.select_dtypes(include="object").columns)

cat_cols = df.select_dtypes(include="object").columns

selected_col = st.selectbox(
    "Select a Categorical Column",
    cat_cols
)

st.subheader(selected_col)

st.dataframe(df[[selected_col]])

# -------------------------------
# Filter by Class
# -------------------------------
mushroom_type = st.selectbox(
    "Filter Mushroom Type",
    ["All", "e", "p"]
)

if mushroom_type == "All":
    filtered_df = df
else:
    filtered_df = df[df["class"] == mushroom_type]

st.dataframe(filtered_df)

# -------------------------------
# Mushroom Distribution
# -------------------------------
st.subheader("Edible vs Poisonous")

fig, ax = plt.subplots()

df["class"].value_counts().plot(kind="bar", ax=ax)

ax.set_xlabel("Class")
ax.set_ylabel("Count")

st.pyplot(fig)

# -------------------------------
# Prepare Data
# -------------------------------
encoded_df = df.copy()

encoders = {}

for col in encoded_df.columns:
    le = LabelEncoder()
    encoded_df[col] = le.fit_transform(encoded_df[col])
    encoders[col] = le

X = encoded_df.drop("class", axis=1)

y = encoded_df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------
st.header("Predict Mushroom Type")

user_input = []

for col in X.columns:

    values = df[col].unique().tolist()

    selected = st.selectbox(col, values)

    encoded = encoders[col].transform([selected])[0]

    user_input.append(encoded)

if st.button("Predict"):

    prediction = model.predict([user_input])

    result = encoders["class"].inverse_transform(prediction)[0]

    if result == "e":
        st.success("✅ Edible Mushroom")
    else:
        st.error("☠️ Poisonous Mushroom")