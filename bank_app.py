import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import plotly.graph_objects as go

# Cargar el modelo entrenado
model = joblib.load('modelo_banco.pkl')

# Obtener las columnas que el modelo espera
x_train = pd.read_csv("model_data/x_train.csv") 
model_feature_names = x_train.columns

# Título
st.title('Predicción de Clientes Potenciales para Depósito Bancario')

# Descripción
st.write("""
    Esta aplicación predice la probabilidad de que un cliente acepte un depósito a plazo fijo.
    Introduce los datos del cliente para realizar la predicción:
""")

# Entrada de datos del cliente
edad = st.number_input('Edad del cliente', min_value=18, max_value=100, value=30)
saldo_bancario = st.number_input('Saldo bancario anual', min_value=0.0, value=10000.0)
empleo = st.selectbox('Trabajo del cliente', ['admin.', 'technician', 'services', 'retired', 'blue-collar', 'entrepreneur', 'housemaid', 'student', 'unemployed', 'management', 'self-employed', 'unknown'])
estado_civil = st.selectbox('Estado civil', ['single', 'married', 'divorced'])
nivel_educacion = st.selectbox('Nivel de Educación', ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox('¿El cliente tiene impagados?', ['yes', 'no'])
vivienda = st.selectbox('¿El cliente tiene vivienda propia?', ['yes', 'no'])
prestamo = st.selectbox('¿El cliente tiene préstamos pendientes?', ['yes', 'no'])
poutcome = st.selectbox('Medio de comunicación con el cliente', ['cellular', 'telephone'])
contact = st.selectbox('Resultado del último contacto', ['success', 'nonexistent', 'failure'])
mes = st.selectbox('Mes del año, del último contacto', ['jan', 'feb', 'mar', 'march', 'jun', 'jul', 'aug', 'oct', 'nov', 'dec'])

# Verificaciones
if saldo_bancario < 0:
    st.error("El saldo bancario no puede ser negativo.")
    saldo_bancario = 0.0
if edad < 18:
    st.error("La edad del cliente debe ser al menos 18 años.")
    edad = 18

# Características numéricas
dia = st.number_input('Día del mes', min_value=1, max_value=31, value=5)
campaña = st.number_input('Número de contactos en la campaña', min_value=0, value=1)
pdays = st.number_input('Número de días desde el último contacto', min_value=-1, value=-1)
previous = st.number_input('Número de contactos previos', min_value=0, value=0)
duration = st.number_input('Duración del último contacto, en segundos', min_value=0, value=10)

# Corregir los valores de cero antes de aplicar logaritmos, para evitar log(0)
saldo_bancario_log = np.log(saldo_bancario + 1e-6)
campaña_log = np.log(campaña + 1e-6)
pdays_log = np.log(pdays + 1e-6) if pdays != -1 else 0  # Tratamos -1 como 0 en este caso
previous_log = np.log(previous + 1e-6)
duration_log = np.log(duration + 1e-6)

# Preparar las características de entrada
entradas = pd.DataFrame([[edad, saldo_bancario_log, empleo, estado_civil, nivel_educacion, default, vivienda, prestamo, 
                          dia, campaña_log, pdays_log, previous_log, duration_log, poutcome, contact, mes]],
                        columns=['age', 'balance_log', 'job', 'marital', 'education', 'default', 'housing', 'loan', 
                                 'day', 'campaign_log', 'pdays_log', 'previous_log', 'duration_log', 'poutcome', 'contact', 'month'])

# 1. Codificar las variables categóricas
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_data = encoder.fit_transform(entradas[categorical_columns])

# Convertimos a DataFrame para concatenar
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# 2. Concatenar con las columnas numéricas
numerical_columns = ['age', 'balance_log', 'day', 'campaign_log', 'pdays_log', 'previous_log', 'duration_log']
entradas_numericas = entradas[numerical_columns]
entradas_transformadas = pd.concat([entradas_numericas, encoded_df], axis=1)

# Verificar que las columnas estén alineadas con las del modelo
model_feature_names = model.feature_names_in_

# Rellenamos las columnas faltantes con 0
missing_columns = [col for col in model_feature_names if col not in entradas_transformadas.columns]
for col in missing_columns:
    entradas_transformadas[col] = 0

entradas_transformadas = entradas_transformadas[model_feature_names]

st.write("""
         Nota: El modelo es sensible a la alta variedad de interacciones entre características. Una pequeño cambio, en un 
        par de variables, apenas afectan a la probabilidad final, a excepción de 'duration_log'; la única variable que por sí sola puede convertir una
         probabilidad en positiva o negativa.
         """)

# Predicción
if st.button('Predecir'):
    try:
        # Realizar la predicción
        prediccion = model.predict(entradas_transformadas)
        probabilidad = model.predict_proba(entradas_transformadas)[:, 1][0] * 100

        # Mostrar los resultados
        if prediccion[0] == 1:
            st.success(f"¡El cliente está interesado en el depósito! Probabilidad: {probabilidad:.2f}%")
        else:
            st.error(f"El cliente no está interesado en el depósito. Probabilidad: {probabilidad:.2f}%")

        # Gráfico circular
        fig = go.Figure(data=[go.Pie(
            labels=['Interesado', 'No Interesado'],
            values=[probabilidad, 100 - probabilidad],
            hole=0.5,
            marker=dict(colors=['#63C132', '#FF3E3E']),
            textinfo='label+percent',
            pull=[0.1, 0]
        )])

        fig.update_layout(
            title="Probabilidad de aceptación del depósito",
            title_x=0.5,
            font=dict(size=16)
        )

        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error en la predicción: {e}")
