"""UI Streamlit: consume la API desplegada en Kubernetes (Service adult-income-api).

Requiere: kubectl port-forward svc/adult-income-api 8080:8000
"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8080")

st.title("Predicción de ingreso (Adult Census)")
st.caption(f"API: {API} (Service de Kubernetes)")

try:
    info = requests.get(f"{API}/info", timeout=5).json()
    st.success(f"Modelo {info['model']} v{info['version']} (alias {info['alias']}) · "
               f"run {info['run_id'][:8]} · pod {info['pod']}")
except Exception as e:
    st.error(f"No se pudo contactar la API: {e}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("age", 17, 90, 39)
    fnlwgt = st.number_input("fnlwgt", 10000, 1500000, 77516)
    edu_num = st.slider("education-num", 1, 16, 13)
    gain = st.number_input("capital-gain", 0, 100000, 2174)
    loss = st.number_input("capital-loss", 0, 5000, 0)
    hours = st.slider("hours-per-week", 1, 99, 40)
with c2:
    workclass = st.selectbox("workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"], index=5)
    education = st.selectbox("education", ["Bachelors", "HS-grad", "Some-college", "Masters", "Doctorate", "Assoc-voc", "11th", "9th"])
    marital = st.selectbox("marital-status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed"])
    occupation = st.selectbox("occupation", ["Adm-clerical", "Exec-managerial", "Prof-specialty", "Craft-repair", "Sales", "Other-service", "Tech-support"])
    relationship = st.selectbox("relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried"])
    race = st.selectbox("race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
    sex = st.selectbox("sex", ["Male", "Female"])
    country = st.selectbox("native-country", ["United-States", "Mexico", "Philippines", "Germany", "Canada", "India"])

if st.button("Predecir"):
    record = {"age": age, "workclass": workclass, "fnlwgt": fnlwgt, "education": education,
              "education-num": edu_num, "marital-status": marital, "occupation": occupation,
              "relationship": relationship, "race": race, "sex": sex, "capital-gain": gain,
              "capital-loss": loss, "hours-per-week": hours, "native-country": country}
    r = requests.post(f"{API}/predict", json={"instances": [record]}, timeout=10)
    if r.ok:
        out = r.json()
        st.metric("Ingreso predicho", ">50K" if out["predictions"][0] == 1 else "<=50K")
        st.caption(f"Atendido por el pod {out['pod']} (modelo v{out['model_version']})")
    else:
        st.error(r.text)
