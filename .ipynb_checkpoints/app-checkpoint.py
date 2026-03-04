import streamlit as st
import numpy as np
import joblib

#-----------Load files---------------
model =joblib.load("rf_model.pkl")
scaler=joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

st.set_page_config(page_title="Dry Eye Prediction", layout="centered")
st.title("Dry Eye Disease Prediction")

st.write("enter patient details:")

#-------------INPUTS----------------

#  Gender
gender_label = st.selectbox("Gender", ["Male","Female"])
gender = "M" if gender_label == "Male" else "F"
gender_encoded = label_encoders["Gender"].transform([gender])[0]

#Numeric Inputs
age = st.number_input("Age",1, 100, 30)
sleep_duration = st.number_input("Sleep duration (hours)",0.0, 24.0, 7.0)
sleep_quality= st.slider("Sleep quality (1-5)",1, 5, 3)
stress_level = st.slider("Stress level (1-5)", 1, 5, 3)
heart_rate = st.number_input("Heart rate", 40, 200, 70)
daily_steps = st.number_input("Daily steps", 0, 50000, 5000)
physical_activity = st.number_input("Physical activity (min/day)", 0, 300, 30)
height = st.number_input("Height (cm)",100, 220, 165)
weight = st.number_input("weight (kg)", 30, 200, 65)
avg_screen_time = st.number_input("Average screen time (hours)", 0.0, 24.0, 6.0)

#Yes/No -  library
def yn(val): return 1 if val== "Yes" else 0

sleep_disorder = yn(st.selectbox("Sleep disorder", ["Yes","No"]))
wake_night = yn(st.selectbox("wake up during night",["Yes","No"]))
sleepy_day = yn(st.selectbox("sleepy_day",["Yes","No"]))
caffeine = yn(st.selectbox("caffeine consumption",["Yes","No"]))
alcohol = yn(st.selectbox("Alcohol Consumption",["Yes","No"]))
smoking = yn(st.selectbox("Smoking",["Yes","No"]))
medical_issue = yn(st.selectbox("Medical Issue",["Yes","No"]))
ongoing_med = yn(st.selectbox("Ongoing medication",["Yes","No"]))
smart_device = yn(st.selectbox("Smart device before bed",["Yes","No"]))
blue_light = yn(st.selectbox("Blue-light filter",["Yes","No"]))
eye_strain= yn(st.selectbox("Discomfort Eye-strain",["Yes","No"]))
redness= yn(st.selectbox("Redness in eye",["Yes","No"]))
itchiness = yn(st.selectbox("Itchiness/ Irritation in eye", ["Yes","No"]))


#------------------FINAL INPUT ORDER---------------------------
input_data = [
    gender_encoded,age,sleep_duration,sleep_quality,stress_level,heart_rate,daily_steps,physical_activity,height ,
    weight,avg_screen_time,sleep_disorder,wake_night,sleepy_day,caffeine,alcohol ,smoking,medical_issue,ongoing_med,
    smart_device,blue_light,eye_strain,redness,itchiness
]

input_array=np.array(input_data).reshape(1, -1)
input_scaled = scaler.transform(input_array)

#-----------Prediction------------
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    result = target_encoder.inverse_transform([pred])[0]

    if result in ["Yes", "Y", 1]:
        st.error(" Dry Eye Disease Detected")
    else:
        st.success(" No Dry Eye Disease Dtetected")

