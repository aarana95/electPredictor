import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

path = r"C:\Users\arana\OneDrive\Documentos\ELECCIONES\cisOctubre.csv"
cis = pd.read_csv(path)

target = cis.INTENCIONG
cis.drop(['INTENCIONG', 'VOTOSIMG', 'CUES', 'P22', 'P27_1', 'P27_2', 'P27_3', 'P27_4', 'P27_5', 'P19_1', 'P19_2', 'P19_3', 'P19_4', 'P19_5'], axis=1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(cis, target, test_size=0.25, random_state=42)

train_mat = xgb.DMatrix(X_train, label=y_train)
test_mat = xgb.DMatrix(X_test, label=y_test)


parametros = {"eta": 0.3, "objective":"multi:softprob", "eval_metric":"mlogloss", "num_class":y_train.nunique()}
rondas = 25

evaluacion = [(test_mat, "eval"), (train_mat, "train")]
modelo = xgb.train(parametros, train_mat, rondas, evaluacion)
importancia = modelo.get_fscore()
{k: v for k, v in sorted(importancia.items(), key=lambda item: item[1], reverse = True)}

#Columnas seleccionadas en base al fscore()

columnas =['ESCIDEOL', 'EDAD', 'ESCIDEOLPOLI_2', 'PROV', 'P17_2',  'P18_2', 'CNO11']
new_pred = cis[columnas]

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_pred, target, test_size=0.25, random_state=42)
train_mat_new = xgb.DMatrix(X_train_new, label=y_train_new)
test_mat_new = xgb.DMatrix(X_test_new, label=y_test_new)

parametros = {"eta": 0.3, "objective":"multi:softprob", "eval_metric":"mlogloss", "num_class":y_train_new.nunique()}
rondas = 25

evaluacion_new = [(test_mat_new, "eval"), (train_mat_new, "train")]

modelo = xgb.train(parametros, train_mat_new, rondas, evaluacion_new)

joblib.dump(modelo, "modelo")