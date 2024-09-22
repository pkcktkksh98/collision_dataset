from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
app = FastAPI()

labels = [
    "Property Damage Only Collision", 
    "Injury Collision", 
    "Unknown", 
    "Serious Injury Collision", 
    "Fatality Collision"
] 
class Features(BaseModel):
    PERSONCOUNT: int
    PEDCOUNT: int
    PEDCYLCOUNT: int
    VEHCOUNT: int
    INJURIES: int
    SERIOUSINJURIES: int
    JUNCTIONTYPE: int
    SDOT_COLCODE: int
    UNDERINFL: int
    LIGHTCOND: int
    PEDROWNOTGRNT: int
    ST_COLCODE: int
    HITPARKEDCAR: int
class Model(BaseModel):
    MODEL : str
@app.get("/ping")
def ping():
    return{"message":"Working!"}

@app.post("/predict/")
def predict(features: Features, model: Model):
    features_input = np.array([features.PERSONCOUNT,
                               features.PEDCOUNT,
                               features.PEDCYLCOUNT,
                               features.VEHCOUNT,
                               features.INJURIES,
                               features.SERIOUSINJURIES,
                               features.JUNCTIONTYPE,
                               features.SDOT_COLCODE,
                               features.UNDERINFL,
                               features.LIGHTCOND,
                               features.PEDROWNOTGRNT,
                               features.ST_COLCODE,
                               features.HITPARKEDCAR]).reshape(1, -1)
    
 
    model  = joblib.load(f'../../model/{model.MODEL}')
    

    try:
        pred = model.predict(features_input)
        pred_labels = labels[pred[0]]
        proba = max(proba[0])   
        return {"prediction": pred_labels, "prediction_score": proba}

    except Exception as e:
        return {"prediction": pred_labels, "error": str(e)}
    ##---------------------
    # # proba = model.predict_proba(features_input)
    # pred_labels = labels[pred[0]]
    # proba = max(proba[0])

    # return{"prediction":pred_labels}
    #     #    "prediction score":proba}
