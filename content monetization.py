# LOAD MODEL & FEATURES
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import base64

model = joblib.load("Regression.pkl")  
feature_columns = model.feature_names_in_

#bacground 
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        height: 100vh;
        width: 100vw;
        opacity: 1.1;
        z-index: -1;
    }}
    </style>
    """, unsafe_allow_html=True)

add_bg_from_local("image.png")

st.title("YouTube Add Revenue Predictor")
st.write("To predict change the details accordingly.")

# USER INPUT FORM

st.subheader("📊 Video Metrics")

views = st.number_input("Views", min_value=0, value=10000)
likes = st.number_input("Likes", min_value=0, value=500)
comments = st.number_input("Comments", min_value=0, value=100)
watch_time = st.number_input("Watch Time (min)", min_value=1.0, value=15000.0)
video_length = st.number_input("Video Length (min)", min_value=1.0, value=10.0)
subscribers = st.number_input(" Subscribers", min_value=0, value=50000)

st.subheader("📱 Audience Information")
# categorical inputs
category = st.selectbox(
        "Category",
        ["Education", "Gaming", "Entertainment", "Music", "Tech"]
    )
device = st.selectbox(
        "Top Device",
        ["Mobile", "TV", "Tablet"]
    )
country = st.selectbox(
        "Top Country",
        ["IN", "US", "UK", "CA", "DE"]
    )

st.subheader("📅 Upload Details")
# date features
upload_year = st.number_input("Upload Year", min_value=200, max_value=2030, value=2024)
upload_month = st.number_input("Upload Month", min_value=1, max_value=12, value=9)
upload_day = st.number_input("Upload Day", min_value=1, max_value=31, value=15)
upload_hour = st.number_input("Upload Hour (0-23)", min_value=0, max_value=23, value=10)
upload_weekday = st.number_input("Upload Weekday (0=Mon, 6=Sun)", min_value=0, max_value=6, value=2)

# PROCESS INPUT INTO MODEL FORMAT

def prepare_input():
    # Base numeric + date features
    input_dict = {
        'views': views,
        'likes': likes,
        'comments': comments,
        'watch_time_minutes': watch_time,
        'video_length_minutes': video_length,
        'subscribers': subscribers,
        'month': upload_month,
        'year': upload_year,
        'upload_year': upload_year,
        'upload_month': upload_month,
        'upload_day': upload_day,
        'upload_hour': upload_hour,
        'upload_weekday': upload_weekday,
    }

    # Create DataFrame FIRST 
    final_df = pd.DataFrame([input_dict])

    # One-hot encode selected category/device/country
    if f"category_{category}" in feature_columns:
        final_df[f"category_{category}"] = 1

    if f"device_{device}" in feature_columns:
        final_df[f"device_{device}"] = 1

    if f"country_{country}" in feature_columns:
        final_df[f"country_{country}"] = 1

    # Add missing columns with 0
    for col in feature_columns:
        if col not in final_df.columns:
            final_df[col] = 0

    # Reorder columns to match training
    final_df = final_df[feature_columns]

    return final_df

# PREDICTION

# Prepare input dataframe
input_df = prepare_input()

# Prediction
if st.button("Predict Ad Revenue"):
    input_df = prepare_input()

    prediction = model.predict(input_df)
   #np.dot(
    #    input_df.values,
    #    coefficients
    #) + intercept

    st.success(f"Estimated Ad Revenue: ${prediction[0]:.2f}")
