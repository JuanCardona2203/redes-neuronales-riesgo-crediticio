import os
import sys

import pandas as pd
import streamlit as st
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import CreditRiskMLP
from src.predict import predecir, VARIABLES

SEED = 42
URL = 'https://raw.githubusercontent.com/pr1m8/Credit-Default/main/credit_risk_dataset.csv'


@st.cache_resource
def cargar_recursos():
    df = pd.read_csv(URL)[VARIABLES + ['loan_status']].copy()
    df = df[df['person_age'] <= 100].copy()
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())
    X_train, _, _, _ = train_test_split(
        df[VARIABLES], df['loan_status'], test_size=0.20, random_state=SEED, stratify=df['loan_status']
    )
    scaler = MinMaxScaler().fit(X_train)

    model = CreditRiskMLP(input_dim=6)
    ruta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'mlp_credit_risk.pt')
    model.load_state_dict(torch.load(ruta))
    model.eval()
    return scaler, model


st.title('Predicción de Riesgo Crediticio (MLP)')
st.write('Ingrese los datos de la solicitud para estimar la probabilidad de mora a 90 días.')

scaler, model = cargar_recursos()

col1, col2 = st.columns(2)
with col1:
    person_age = st.number_input('Edad', min_value=18, max_value=100, value=30)
    person_income = st.number_input('Ingreso anual (USD)', min_value=0, value=50000)
    loan_amnt = st.number_input('Monto del préstamo (USD)', min_value=0, value=10000)
with col2:
    loan_int_rate = st.number_input('Tasa de interés (%)', min_value=0.0, value=11.0)
    loan_percent_income = st.number_input('Proporción cuota/ingreso', min_value=0.0, max_value=1.0, value=0.2)
    cb_hist = st.number_input('Años de historial crediticio', min_value=0, value=5)

if st.button('Evaluar solicitud'):
    fila = pd.DataFrame([[person_age, person_income, loan_amnt,
                          loan_int_rate, loan_percent_income, cb_hist]], columns=VARIABLES)
    prob, pred = predecir(model, scaler.transform(fila))
    st.metric('Probabilidad de mora', f'{prob[0]*100:.1f}%')
    if pred[0] == 1:
        st.error('Decisión sugerida: DENEGAR')
    else:
        st.success('Decisión sugerida: APROBAR')
