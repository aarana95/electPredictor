import joblib
import streamlit as st
import numpy as np
import xgboost as xgb
import pandas as pd
model = joblib.load("modelo.json")

respuestas = list()
st.title('Predictor electoral')

respuestas.append(st.slider('Escala de autoubicación ideologica: 1 iz - 10 dr', 1, 10, 1))
respuestas.append(st.number_input('Introduce tu edad:', step=1))
respuestas.append(st.slider('Donde ubicaría la escala de ubicación ideologica de Pablo Casado:', 1, 10, 1))



efectos = {'Los efectos sobre la salud':1,'Los efectos sobre la economía y el empleo':2, 'Ambos por igual':3, 'Ni unos ni otros':4}
respuestas.append(efectos[st.selectbox("En estos momentos, ¿qué le preocupa a Ud. más, los efectos de esta crisis sobre la salud, o los efectos de la crisis sobre la economía y el empleo?",
                                       options=list(efectos.keys()))])



esp_actual = {'Muy Buena':1, 'Buena':2, 'Regular':3, 'Mala':4, 'Muy mala':5}
respuestas.append(esp_actual[st.selectbox("""Refiriéndonos a la situación económica general de España actualmente, ¿cómo la calificaría Ud.: muy buena, buena, mala o muy mala?""",
                                       options=list(esp_actual.keys()))])


respuestas.append(esp_actual[st.selectbox("""¿cómo la calificaría Ud. su situación económica personal: muy buena, buena, mala o muy mala?""",
                                       options=list(esp_actual.keys()))])

problemas = {'Los peligros para la salud: COVID-19':95, 'La crisis económica, los problemas de índole económica':8,
'El paro':1, 'Los problemas políticos en general':51, 'Lo que hacen los partidos políticos':50,
'El mal comportamiento de los/as políticos/as':13,
'La corrupción y el fraude':11, 'La sanidad':6, 'Los problemas de índole social':16,
'La inmigración':18,'Los problemas relacionados con la calidad del empleo':9,
'La independencia de Cataluña':45,'Las pensiones':12,'La violencia de género':19,'La falta de acuerdos':93,
'Poca conciencia ciudadana':94, 'Ninguno':97}

#respuestas.append(problemas[st.selectbox('Cual es el principal problema que tiene españa:', options=list(problemas.keys()))])
#respuestas.append(problemas[st.selectbox('Y el segundo:', options=list(problemas.keys()))])


problemas_personales = {'Los peligros para la salud: COVID-19':95, 'La crisis económica, los problemas de índole económica':8,
             'Avituallamiento de víveres en el hogar':94, 'Tener que estar enclaustrado/a en casa':93, 'La educacion':22,
             'Las preocupaciones y situaciones personales':29, 'Los problemas relacionados con la juventud':20,
'El paro':1, 'Los problemas políticos en general':51, 'Lo que hacen los partidos políticos':50,
'El mal comportamiento de los/as políticos/as':13,
'La corrupción y el fraude':11, 'La sanidad':6, 'Los problemas de índole social':16,
'La inmigración':18,'Los problemas relacionados con la calidad del empleo':9,
'La independencia de Cataluña':45,'Las pensiones':12,'La violencia de género':19,'La falta de acuerdos':93,
'Poca conciencia ciudadana':94, 'Ninguno':97}

#respuestas.append(problemas_personales[st.selectbox('Cual es el principal problema que tiene usted:', options=list(problemas_personales.keys()))])
#respuestas.append(problemas_personales[st.selectbox('Y el segundo:', options=list(problemas_personales.keys()))])


ocupacion = {'Directores/as y gerentes':1, 'Profesionales y científicos/as e intelectuales':2, 'Técnicos/as y profesionales de nivel medio':3,
'Personal de apoyo administrativo':4, 'Trabajadores/as de los servicios y vendedores/as de comercios y mercados':5,
'Agricultores/as y trabajadores/as cualificados/as agropecuarios/as, forestales y pesqueros/as':6,
'Oficiales/as, operarios/as y artesanos/as de artes mecánicas y de otros oficios':7,
'Operadores/as de instalaciones y máquinas y ensambladores/as':8,
'Ocupaciones elementales':9, 'Ocupaciones militares y cuerpos policiales':10, 'Otra/o':11, 'N.C.':99}
respuestas.append(ocupacion[st.selectbox('¿Me puede decir cuál es su ocupación actual?:', options=list(ocupacion.keys()))])



relg = {"católico/a practicante":1, "católico/a no practicante":2, "creyente de otra religión":3, "agnóstico/a":4, "indiferente o no creyente":5, "ateo/a":6}
respuestas.append(relg[st.selectbox("¿Cómo se define Ud. en materia religiosa?", options=list(relg.keys()))])

columnas =['ESCIDEOL', 'EDAD', 'ESCIDEOLPOLI_2', 'P2', 'P15', 'P16', 'CNO11', 'RELIGION']
obs = pd.DataFrame(columns = columnas)
obs.loc[0] = respuestas
obs = obs.astype('int32')
prediccion = model.predict(xgb.DMatrix(obs))
st.subheader("Prediccion:")




import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['PP', 'PSOE', 'PODEMOS', 'CIUDADANOS', 'VOX', 'NO VOTA']
sizes = prediccion[0]
#explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
colores = ['b', 'r', 'purple', 'orange', 'lawngreen', 'black']
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', #explode=explode,
        shadow=True, startangle=90, colors = colores)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)