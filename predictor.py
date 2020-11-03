import joblib
import streamlit as st

xgb = joblib.load("modelo")

respuestas = list()
st.title('Predictor electoral')

respuestas.append(st.slider('Escala de autoubicación ideologica: 1 iz - 10 dr', 1, 10, 1))
respuestas.append(st.number_input('Introduce tu edad:', format='%d'))
respuestas.append(st.slider('Donde ubicaría la escala de ubicación ideologica de Pablo Casado:', 1, 10, 1))

problemas = {'Los peligros para la salud: COVID-19':95, 'La crisis económica, los problemas de índole económica':8,
'El paro':1, 'Los problemas políticos en general':51, 'Lo que hacen los partidos políticos':50,
'El mal comportamiento de los/as políticos/as':13,
'La corrupción y el fraude':11, 'La sanidad':6, 'Los problemas de índole social':16,
'La inmigración':18,'Los problemas relacionados con la calidad del empleo':9,
'La independencia de Cataluña':45,'Las pensiones':12,'La violencia de género':19,'La falta de acuerdos':93,
'Poca conciencia ciudadana':94, 'Ninguno':97}

aa = st.select_slider('Cual es el principal problema que tiene españa:', options=list(problemas.keys()))
st.write(respuestas)
st.write(aa)