import streamlit as st
import joblib
import pandas as pd

# Load the comprehensive pipeline model
model = joblib.load('House_Price.pkl')

st.title("🏠 Advanced House Price Predictor")
st.write("Provide the structural specifications below to estimate market value.")

# Create an organized 2-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sq ft)", min_value=100, max_value=20000, value=1500, step=50)
    bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
    bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4, 5], index=1)
    floors = st.selectbox("Floors", options=[1, 2, 3, 4], index=0)

with col2:
    year_built = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000, step=1)
    location = st.selectbox("Location", options=["Downtown", "Suburban", "Rural"])
    condition = st.selectbox("Condition", options=["Excellent", "Good", "Fair", "Poor"])
    garage = st.selectbox("Has Garage?", options=["Yes", "No"])

# Predict execution
if st.button("Predict House Price", type="primary"):
    
    # 1. Store inputs in a DataFrame matching the original training column headers exactly
    input_data = pd.DataFrame([{
        'Area': area,
        'Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'YearBuilt': year_built,
        'Location': location,
        'Condition': condition,
        'Garage': garage
    }])
    
    # 2. Pipeline processes and predicts automatically
    prediction = model.predict(input_data)
    predicted_price = prediction.item()
    
    # 3. Output response
    st.success(f"### Estimated Price: ${predicted_price:,.2f}")
