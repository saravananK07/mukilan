import streamlit as st
import numpy as np
import pickle

# Correct file paths
model_path = 'model.pkl'
scaler_path = 'muksclr.pkl'

# Load the model and scaler
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

def predict(features):
    features = np.array([features])
    features = scaler.transform(features)  # Normalize the features
    prediction = model.predict(features).flatten()
    return prediction

# Streamlit application
st.title('Regression Model Deployment')
st.write("Enter the features to get predictions:")

# Input fields for features
area = st.number_input('Area', min_value=0.0, format="%.2f", help="Area of the sensor field.")
sensing_range = st.number_input('Sensing Range', min_value=0.0, format="%.2f", help="Sensing range of the sensors.")
transmission_range = st.number_input('Transmission Range', min_value=0.0, format="%.2f", help="Transmission range of the sensors.")
num_sensor_nodes = st.number_input('Number of Sensor Nodes', min_value=0, format="%d", help="Number of sensor nodes in the field.")

# Button to make prediction
if st.button('Predict'):
    features = [area, sensing_range, transmission_range, num_sensor_nodes]
    prediction = predict(features)
    st.write(f'Predicted Number of Barriers: {prediction[0]:.2f}')
