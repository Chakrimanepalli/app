# -*- coding: utf-8 -*-
"""

@author: chakri-PC
"""

import streamlit as st
import pandas as pd
import numpy as np
pip install xgboost
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from scipy import stats

# Load data
filepath = "C:\\Users\\chakri\\Downloads\\energy_production.csv"
df = pd.read_csv(filepath, sep=';')

# Remove outliers using Z-score
z_scores_r_humidity = np.abs(stats.zscore(df['r_humidity']))
z_scores_amb_pressure = np.abs(stats.zscore(df['amb_pressure']))
threshold = 3
df_cleaned = df[(z_scores_r_humidity < threshold) & (z_scores_amb_pressure < threshold)]

# Prepare data
X = df_cleaned[['temperature', 'exhaust_vacuum', 'amb_pressure', 'r_humidity']]
y = df_cleaned['energy_production']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)

# Streamlit app
st.title('Energy Production Prediction')

# Sidebar for user input
st.sidebar.title('Input Parameters')
temperature = st.sidebar.slider('Temperature', min_value=df['temperature'].min(), max_value=df['temperature'].max())
exhaust_vacuum = st.sidebar.slider('Exhaust Vacuum', min_value=df['exhaust_vacuum'].min(), max_value=df['exhaust_vacuum'].max())
amb_pressure = st.sidebar.slider('Ambient Pressure', min_value=df['amb_pressure'].min(), max_value=df['amb_pressure'].max())
r_humidity = st.sidebar.slider('Relative Humidity', min_value=df['r_humidity'].min(), max_value=df['r_humidity'].max())

# Predict button
if st.sidebar.button('Predict'):
    # Predict energy production
    input_data = np.array([[temperature, exhaust_vacuum, amb_pressure, r_humidity]])
    prediction = xgb_model.predict(input_data)[0]
    st.write(f'Predicted Energy Production: {prediction:.2f} MW')


if __name__ == '__main__':
    main()
    
