"""
mystreamlit.py — Despliegue del modelo final de predicción de depósito bancario
Práctica 1 — Aprendizaje Automático 2025-26
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="Predicción de Depósito Bancario",
    page_icon="🏦",
    layout="centered",
)

# ─── Carga del modelo ─────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load("modelo_final.joblib")
        return modelo
    except FileNotFoundError:
        try:
            modelo = joblib.load("modelo_final.pkl")
            return modelo
        except FileNotFoundError:
            return None

modelo = cargar_modelo()

# ─── Cabecera ─────────────────────────────────────────────────────────────────
st.title("🏦 Predicción de Suscripción a Depósito")
st.markdown(
    "Introduce los datos del cliente para predecir si suscribirá un **depósito a plazo**."
)

if modelo is None:
    st.error(
        "⚠️ No se encontró el modelo (`modelo_final.joblib` o `modelo_final.pkl`). "
        "Asegúrate de que el archivo está en el mismo directorio que este script."
    )
    st.stop()

# ─── Opciones de las variables categóricas ────────────────────────────────────
JOBS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]
MARITAL   = ["divorced", "married", "single"]
EDUCATION = ["primary", "secondary", "tertiary", "unknown"]
DEFAULT   = ["no", "yes"]
HOUSING   = ["no", "yes"]
LOAN      = ["no", "yes"]
CONTACT   = ["cellular", "telephone", "unknown"]
MONTHS    = ["jan", "feb", "mar", "apr", "may", "jun",
             "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME  = ["failure", "other", "success", "unknown"]

# ─── Formulario de entrada ────────────────────────────────────────────────────
st.subheader("Datos del cliente")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Perfil personal**")
    age       = st.number_input("Edad (años)", min_value=18, max_value=100, value=40)
    job       = st.selectbox("Tipo de trabajo", JOBS)
    marital   = st.selectbox("Estado marital", MARITAL)
    education = st.selectbox("Nivel de educación", EDUCATION)

    st.markdown("**Situación financiera**")
    default = st.selectbox("¿Tiene crédito impagado?", DEFAULT)
    balance = st.number_input("Balance anual medio (€)", value=1000, step=100)
    housing = st.selectbox("¿Tiene hipoteca?", HOUSING)
    loan    = st.selectbox("¿Tiene préstamo personal?", LOAN)

with col2:
    st.markdown("**Último contacto**")
    contact     = st.selectbox("Tipo de contacto", CONTACT)
    month       = st.selectbox("Mes del último contacto", MONTHS)
    day_of_week = st.number_input("Día del mes del último contacto", min_value=1, max_value=31, value=15)
    duration    = st.number_input("Duración del último contacto (segundos)", min_value=0, value=200, step=10)

    st.markdown("**Campaña**")
    campaign = st.number_input("Nº de contactos en esta campaña", min_value=1, value=2)
    pdays    = st.number_input(
        "Días desde último contacto anterior (-1 = nunca contactado)",
        min_value=-1, value=-1
    )
    previous = st.number_input("Nº de contactos en campañas anteriores", min_value=0, value=0)
    poutcome = st.selectbox("Resultado de campaña anterior", POUTCOME)

# ─── Preproceso de pdays (igual que en el notebook) ───────────────────────────
def preparar_instancia(age, job, marital, education, default, balance,
                       housing, loan, contact, day_of_week, month, duration,
                       campaign, pdays, previous, poutcome):
    was_contacted_before = int(pdays != -1)
    pdays_proc = float("nan") if pdays == -1 else float(pdays)

    data = {
        "age":                  age,
        "job":                  job,
        "marital":              marital,
        "education":            education,
        "default":              default,
        "balance":              balance,
        "housing":              housing,
        "loan":                 loan,
        "contact":              contact,
        "day":                  day_of_week,
        "month":                month,
        "duration":             duration,
        "campaign":             campaign,
        "pdays":                pdays_proc,
        "previous":             previous,
        "poutcome":             poutcome,
        "was_contacted_before": was_contacted_before,
    }
    return pd.DataFrame([data])

# ─── Predicción ───────────────────────────────────────────────────────────────
st.divider()

if st.button("🔍 Predecir", use_container_width=True, type="primary"):
    X_nuevo = preparar_instancia(
        age, job, marital, education, default, balance,
        housing, loan, contact, day_of_week, month, duration,
        campaign, pdays, previous, poutcome
    )

    prediccion = modelo.predict(X_nuevo)[0]

    st.subheader("Resultado")

    if prediccion == "yes":
        st.success("✅ El cliente **SÍ suscribirá** el depósito")
    else:
        st.error("❌ El cliente **NO suscribirá** el depósito")

    st.metric("Predicción", prediccion.upper())

    # ── Detalle de la instancia introducida ──
    with st.expander("Ver datos enviados al modelo"):
        X_nuevo_show = X_nuevo.copy()
        X_nuevo_show["pdays"] = pdays  # mostrar el valor original
        st.dataframe(X_nuevo_show, use_container_width=True)

# ─── Predicción por lote (verificación pipeline vs app) ───────────────────────
st.divider()
st.subheader("Verificación: comparar pipeline vs app")
st.markdown(
    "Introduce a continuación dos instancias manualmente para demostrar que las "
    "predicciones del pipeline (notebook) y la app Streamlit coinciden."
)

with st.expander("📋 Instancias de verificación"):
    st.markdown("**Instancia 1**")
    inst1 = preparar_instancia(
        age=45, job="management", marital="married", education="tertiary",
        default="no", balance=5000, housing="yes", loan="no",
        contact="cellular", day_of_week=15, month="may", duration=400,
        campaign=2, pdays=-1, previous=0, poutcome="unknown"
    )
    pred1 = modelo.predict(inst1)[0]

    st.dataframe(inst1, use_container_width=True)
    st.write(f"➡️ Predicción: **{pred1}")

    st.markdown("**Instancia 2**")
    inst2 = preparar_instancia(
        age=30, job="student", marital="single", education="secondary",
        default="no", balance=200, housing="no", loan="no",
        contact="telephone", day_of_week=8, month="nov", duration=120,
        campaign=5, pdays=90, previous=1, poutcome="failure"
    )
    pred2 = modelo.predict(inst2)[0]

    st.dataframe(inst2, use_container_width=True)
    st.write(f"➡️ Predicción: **{pred2}")

    st.info(
        "Copia estas dos instancias en tu notebook y comprueba que `modelo_produccion.predict()` "
        "devuelve los mismos resultados para incluirlo en el PDF de verificación."
    )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Práctica 1 — Aprendizaje Automático 2025-26 · UC3M")