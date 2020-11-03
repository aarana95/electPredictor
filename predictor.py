import joblib
import streamlit as st

xgb = joblib.load("modelo")

respuestas = list()
st.title('Predictor electoral')

respuestas.append(st.slider('Escala de autoubicación ideologica:', 1, 10, 1))
respuestas.append(st.number_input('Introduce tu edad:'))
st.write(respuestas)
