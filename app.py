import streamlit as st
import joblib
import pandas as pd

pipeline = joblib.load("flight_price_pipeline.pkl")

st.title("✈️ Flight Ticket Price Prediction")

# User Inputs
airline = st.selectbox(
    "Airline",
    ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Vistara"]
)

source = st.selectbox(
    "Source",
    ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore", "Kochi", "Hybrabad"]
)

destination = st.selectbox(
    "Destination",
    ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore"]
)

total_stops = st.selectbox(
    "Total Stops",
    [0, 1, 2, 3]
)

duration = st.number_input(
    "Duration (Minutes)",
    min_value=30
)

journey_day = st.slider(
    "Journey Day",
    1, 31, 15
)

journey_month = st.slider(
    "Journey Month",
    1, 12, 6
)

# Prediction Button
if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "Airline": [airline],
        "Source": [source],
        "Destination": [destination],
        "Total_Stops": [total_stops],
        "Duration_Mins": [duration],
        "Journey_Day": [journey_day],
        "Journey_Month": [journey_month]
    })

    prediction = pipeline.predict(input_data)

    st.success(f"💰 Predicted Ticket Price: ₹{prediction[0]:,.2f}")