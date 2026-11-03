"""UI Streamlit: consume la API desplegada en Kubernetes (Service adult-income-api).

Requiere: kubectl port-forward svc/adult-income-api 8080:8000
Las etiquetas se muestran en español; a la API se envían los valores originales del dataset.
"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8080")

# valor que espera el modelo -> texto en español
TRABAJO = {"Private": "Empresa privada", "Self-emp-not-inc": "Independiente (sin empresa constituida)",
           "Self-emp-inc": "Empresario (empresa constituida)", "Federal-gov": "Gobierno federal",
           "Local-gov": "Gobierno local", "State-gov": "Gobierno estatal"}
EDUCACION = {"9th": "Secundaria incompleta (9.º)", "11th": "Secundaria incompleta (11.º)",
             "HS-grad": "Bachillerato completo", "Some-college": "Universidad sin terminar",
             "Assoc-voc": "Técnico / vocacional", "Bachelors": "Licenciatura",
             "Masters": "Maestría", "Doctorate": "Doctorado"}
CIVIL = {"Never-married": "Soltero/a", "Married-civ-spouse": "Casado/a", "Divorced": "Divorciado/a",
         "Separated": "Separado/a", "Widowed": "Viudo/a"}
OCUPACION = {"Adm-clerical": "Administrativo", "Exec-managerial": "Directivo / gerente",
             "Prof-specialty": "Profesional especializado", "Craft-repair": "Oficios / reparaciones",
             "Sales": "Ventas", "Other-service": "Otros servicios", "Tech-support": "Soporte técnico"}
FAMILIA = {"Not-in-family": "Sin familiares en el hogar", "Husband": "Esposo", "Wife": "Esposa",
           "Own-child": "Hijo/a", "Unmarried": "Pareja sin casar"}
RAZA = {"White": "Blanca", "Black": "Negra", "Asian-Pac-Islander": "Asiática / Islas del Pacífico",
        "Amer-Indian-Eskimo": "Indígena americana / esquimal", "Other": "Otra"}
SEXO = {"Male": "Hombre", "Female": "Mujer"}
PAIS = {"United-States": "Estados Unidos", "Mexico": "México", "Philippines": "Filipinas",
        "Germany": "Alemania", "Canada": "Canadá", "India": "India"}


def elegir(etiqueta, opciones, indice=0, ayuda=None):
    return st.selectbox(etiqueta, list(opciones), index=indice,
                        format_func=opciones.get, help=ayuda)


st.title("Predicción de ingresos")
st.write("Indica los datos de una persona y el modelo estima si gana **más de 50.000 USD al año**. "
         "El modelo se ejecuta en un clúster de Kubernetes.")

try:
    info = requests.get(f"{API}/info", timeout=5).json()
    st.success(f"Conectado · modelo `{info['model']}` versión {info['version']} "
               f"(alias {info['alias']}) · run {info['run_id'][:8]} · pod {info['pod']}")
except Exception as e:
    st.error(f"No se pudo conectar con la API. ¿Está corriendo `kubectl port-forward`? Detalle: {e}")
    st.stop()

c1, c2 = st.columns(2)
with c1:
    st.subheader("Datos numéricos")
    age = st.number_input("Edad (años)", 17, 90, 39)
    edu_num = st.slider("Años de estudio", 1, 16, 13,
                        help="Escala del censo: 9 = bachillerato, 13 = licenciatura, 16 = doctorado.")
    hours = st.slider("Horas de trabajo por semana", 1, 99, 40)
    gain = st.number_input("Ganancias por inversiones (USD/año)", 0, 100000, 2174,
                           help="Ingresos por inversiones o venta de bienes. Si no tiene, pon 0.")
    loss = st.number_input("Pérdidas por inversiones (USD/año)", 0, 5000, 0,
                           help="Pérdidas por inversiones o venta de bienes. Si no tiene, pon 0.")
    fnlwgt = st.number_input(
        "Peso estadístico del censo (opcional)", 10000, 1500000, 77516,
        help="Se llama 'fnlwgt' en el dataset: cuántas personas de la población representa este "
             "registro en el censo. No describe a la persona y casi no influye; déjalo como está.")
with c2:
    st.subheader("Datos de la persona")
    workclass = elegir("Tipo de empleo", TRABAJO, 5)
    education = elegir("Nivel de estudios", EDUCACION, 5,
                       "Debe ser coherente con los años de estudio de la izquierda.")
    marital = elegir("Estado civil", CIVIL)
    occupation = elegir("Ocupación", OCUPACION)
    relationship = elegir("Relación en el hogar", FAMILIA)
    race = elegir("Origen étnico", RAZA)
    sex = elegir("Sexo", SEXO)
    country = elegir("País de origen", PAIS)

if st.button("Predecir", type="primary"):
    record = {"age": age, "workclass": workclass, "fnlwgt": fnlwgt, "education": education,
              "education-num": edu_num, "marital-status": marital, "occupation": occupation,
              "relationship": relationship, "race": race, "sex": sex, "capital-gain": gain,
              "capital-loss": loss, "hours-per-week": hours, "native-country": country}
    try:
        r = requests.post(f"{API}/predict", json={"instances": [record]}, timeout=10)
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        st.stop()
    if r.ok:
        out = r.json()
        if out["predictions"][0] == 1:
            st.success("Resultado: gana **más de 50.000 USD al año**")
        else:
            st.info("Resultado: gana **50.000 USD al año o menos**")
        st.caption(f"Respondió el pod `{out['pod']}` con el modelo versión {out['model_version']}.")
    else:
        st.error(f"La API rechazó la petición: {r.text}")
