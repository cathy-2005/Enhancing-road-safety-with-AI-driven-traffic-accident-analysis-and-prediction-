import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.title("Road Safety Accident Prediction")

# Input form
with st.form("prediction_form"):
    state = st.selectbox("State Name", ['Meghalaya', 'Uttar Pradesh', 'Tamil Nadu'])
    city = st.text_input("City Name", "Unknown")
    year = st.number_input("Year", min_value=2000, max_value=2030, value=2018)
    month = st.selectbox("Month", ['January', 'February', 'March'])
    day = st.selectbox("Day of Week", ['Monday', 'Tuesday'])
    time = st.text_input("Time of Day", "12:00")
    severity = st.selectbox("Accident Severity", ['Minor', 'Major'])
    num_vehicles = st.number_input("Number of Vehicles Involved", 1, 10, 2)
    vehicle_type = st.selectbox("Vehicle Type", ['Car', 'Truck'])
    casualties = st.number_input("Number of Casualties", 0, 100, 0)
    fatalities = st.number_input("Number of Fatalities", 0, 100, 0)
    weather = st.selectbox("Weather Conditions", ['Clear', 'Rainy'])
    road_type = st.selectbox("Road Type", ['National Highway', 'Urban Road'])
    road_condition = st.selectbox("Road Condition", ['Dry', 'Wet'])
    lighting = st.selectbox("Lighting Conditions", ['Daylight', 'Night'])
    traffic_control = st.selectbox("Traffic Control Presence", ['Signs', 'Signals'])
    speed = st.number_input("Speed Limit (km/h)", 0, 200, 50)
    driver_age = st.number_input("Driver Age", 16, 100, 18)
    driver_gender = st.selectbox("Driver Gender", ['Male', 'Female'])
    license_status = st.selectbox("Driver License Status", ['Valid', 'Invalid'])
    alcohol = st.selectbox("Alcohol Involvement", ['Yes', 'No'])
    location_detail = st.selectbox("Accident Location Details", ['Straight Road', 'Curve'])

    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = {
        'State Name': [state],
        'City Name': [city],
        'Year': [year],
        'Month': [month],
        'Day of Week': [day],
        'Time of Day': [time],
        'Accident Severity': [severity],
        'Number of Vehicles Involved': [num_vehicles],
        'Vehicle Type Involved': [vehicle_type],
        'Number of Casualties': [casualties],
        'Number of Fatalities': [fatalities],
        'Weather Conditions': [weather],
        'Road Type': [road_type],
        'Road Condition': [road_condition],
        'Lighting Conditions': [lighting],
        'Traffic Control Presence': [traffic_control],
        'Speed Limit (km/h)': [speed],
        'Driver Age': [driver_age],
        'Driver Gender': [driver_gender],
        'Driver License Status': [license_status],
        'Alcohol Involvement': [alcohol],
        'Accident Location Details': [location_detail]
    }

    input_df = pd.DataFrame(input_data)

    # Encode using saved encoders
    def safe_transform(encoder, values):
        result = []
        for val in values:
            if val in encoder.classes_:
                result.append(encoder.transform([val])[0])
            else:
                result.append(-1)
        return np.array(result)

    encoder_dir = "encoders"
    for column in input_df.columns:
        encoder_path = os.path.join(encoder_dir, f"{column}_encoder.pkl")
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                le = pickle.load(f)
            input_df[column] = safe_transform(le, input_df[column].astype(str))

    st.write("Processed Input Data:")
    st.dataframe(input_df)

    # Dummy prediction for now
    st.success("Prediction: Safe (This is a placeholder)")
