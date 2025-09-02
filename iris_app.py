import numpy as np
import joblib
import streamlit as st

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")
st.title("🌸 Iris Flower Classifier")

model = joblib.load("best_iris_model.joblib")
feature_names = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
species = np.array(["setosa", "versicolor", "virginica"])

cols = st.columns(2)
with cols[0]:
    sl = st.number_input(feature_names[0], 4.0, 8.5, 5.1, step=0.1)
    pl = st.number_input(feature_names[2], 1.0, 7.0, 1.4, step=0.1)
with cols[1]:
    sw = st.number_input(feature_names[1], 2.0, 4.5, 3.5, step=0.1)
    pw = st.number_input(feature_names[3], 0.1, 3.0, 0.2, step=0.1)

if st.button("Predict"):
    x = np.array([[sl, sw, pl, pw]])
    pred = model.predict(x)[0]
    proba = getattr(model, "predict_proba", lambda X: None)(x)
    st.success(f"Prediction: **{species[pred].title()}**")

    if proba is not None:
        st.write("Class probabilities:")
        st.write({species[i].title(): float(p) for i, p in enumerate(proba[0])})
