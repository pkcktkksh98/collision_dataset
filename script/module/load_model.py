import joblib
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

def load_model(data):
   knn=joblib.load("model/knn.pkl")
   mlp=joblib.load("model/mlp.pkl")
   xgb=joblib.load("model/xgb.pkl")

   feature1 = data[[ 'PERSONCOUNT', 'PEDCOUNT', 'PEDCYLCOUNT', 'VEHCOUNT',
       'INJURIES', 'SERIOUSINJURIES', 'JUNCTIONTYPE', 'SDOT_COLCODE',
       'UNDERINFL', 'LIGHTCOND', 'PEDROWNOTGRNT', 'ST_COLCODE',
       'HITPARKEDCAR']]
   label1 = data['SEVERITYCODE']

   x_train, x_test, y_train, y_test = train_test_split(feature1,label1,
                                                      train_size=0.8,
                                                      shuffle=True,
                                                      random_state=42)
   
   knn_score = knn.score(x_test,y_test)
   xgb_score = xgb.score(x_test,y_test)
   # mlp_score = mlp.accuracy_score(x_test,y_test)
   
   return knn_score,xgb_score
