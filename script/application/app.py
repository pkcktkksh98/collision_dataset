import streamlit as st
import joblib
import numpy as np

# Load the model
page = st.selectbox("Select a model:",["knn.pkl","xgb.pkl", "mlp.pkl"])
model = joblib.load(f'../../model/{page}')  # Update with your actual model name
labels = [
    "Property Damage Only Collision", 
    "Injury Collision", 
    "Unknown", 
    "Serious Injury Collision", 
    "Fatality Collision"
] 

st.title("Incident Classification - Prediction APP")
st.header(f'Model: {page}')

# Input sliders for each feature
PERSONCOUNT = int(st.slider("Person Count:", min_value=0, max_value=93, value=0, step=1))
PEDCOUNT = int(st.slider("Pedestrian Count:", min_value=0, max_value=6, value=0, step=1))
PEDCYLCOUNT = int(st.slider("Pedestrian Cyclist Count:", min_value=0, max_value=2, value=0, step=1))
VEHCOUNT = int(st.slider("Vehicle Count:", min_value=0, max_value=15, value=0, step=1))
INJURIES = int(st.slider("Injuries:", min_value=0, max_value=78, value=0, step=1))
SERIOUSINJURIES = int(st.slider("Serious Injuries:", min_value=0, max_value=41, value=0, step=1))
JUNCTIONTYPE = int(st.slider("Junction Type:", min_value=0, max_value=5, value=7, step=1))
SDOT_COLCODE = int(st.slider("SDOT Collision Code:", min_value=0, max_value=39, value=50, step=1))
UNDERINFL = int(st.slider("Under Influence:", min_value=0, max_value=4, value=0, step=1))
LIGHTCOND = int(st.slider("Light Condition:", min_value=0, max_value=9, value=0, step=1))
PEDROWNOTGRNT = int(st.slider("Pedestrian Right of Way Not Granted (0: No, 1: Yes):", min_value=0, max_value=1, value=0, step=1))
ST_COLCODE = int(st.slider("State Collision Code:", min_value=0, max_value=63, value=50, step=1))
HITPARKEDCAR = int(st.slider("Hit Parked Car (0: No, 1: Yes):", min_value=0, max_value=1, value=0, step=1))

# Prediction button
button = st.button("Prediction")

if button:
    # Make prediction
    features_input = np.array([PERSONCOUNT, PEDCOUNT, PEDCYLCOUNT, VEHCOUNT, INJURIES, SERIOUSINJURIES, 
                               JUNCTIONTYPE, SDOT_COLCODE, UNDERINFL, LIGHTCOND, PEDROWNOTGRNT, 
                               ST_COLCODE, HITPARKEDCAR]).reshape(1, -1)
    pred = model.predict(features_input)
    pred_label = labels[pred[0]]
    
    st.subheader(f'Class prediction: {pred_label}')
    

