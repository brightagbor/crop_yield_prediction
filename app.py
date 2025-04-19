import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go


# Load the label-encoded trained model
model = joblib.load('crop_yield_model_label_encoded.pkl')

# Define the label mappings based on your training data
region_mapping = {'East': 0, 'North': 1, 'South': 2, 'West': 3}
soil_mapping = {'Chalky': 0, 'Clay': 1,  'Loamy': 2, 'Peaty': 3, 'Sandy': 4, 'Silt': 5}
crop_mapping = {'Barley': 0, 'Cotton': 1, 'Maize': 2, 'Rice': 3, 'Soybean': 4, 'Wheat': 5}
weather_mapping = {'Sunny': 0, 'Rainy': 1, 'Cloudy': 2}

## Custom app title
st.title("🌾🌾🌾 A Machine Learning App for Crop Yield Prediction Using Weather Data")

def get_user_input():
    with st.sidebar:
        st.markdown("## Input Parameters") # This acts as your side header
        st.markdown("---") # Optional separator

        region_selected = st.selectbox("Region", list(region_mapping.keys()))
        soil_selected = st.selectbox("Soil Type", list(soil_mapping.keys()))
        crop_selected = st.selectbox("Crop", list(crop_mapping.keys()))
        rainfall = st.slider("Rainfall (mm)", 0.0, 1500.0, 100.0)
        temperature = st.slider("Temperature (°C)", 0.0, 100.0, 50.0)
        fertilizer = st.selectbox("Fertilizer Used", ['Yes', 'No'])
        irrigation = st.selectbox("Irrigation Used", ['Yes', 'No'])
        weather_selected = st.selectbox("Weather Condition", list(weather_mapping.keys()))
        days_to_harvest = st.slider("Days to Harvest", 50, 200, 120)

    input_data = pd.DataFrame({
        'Region': [region_mapping[region_selected]], # Use the mapping for the model
        'Soil_Type': [soil_mapping[soil_selected]],
        'Crop': [crop_mapping[crop_selected]],
        'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temperature],
        'Fertilizer_Used': [1 if fertilizer == 'Yes' else 0],
        'Irrigation_Used': [1 if irrigation == 'Yes' else 0],
        'Weather_Condition': [weather_mapping[weather_selected]],
        'Days_to_Harvest': [days_to_harvest]
    })

    # Ensure the column order matches the training data's order after encoding
    input_data = input_data[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
                             'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest']]

    # Store the original selected values for display
    input_data.original_region = region_selected
    input_data.original_soil = soil_selected
    input_data.original_crop = crop_selected
    input_data.original_weather = weather_selected
    return input_data

input_df = get_user_input()

print(input_df)

# Predict
prediction = model.predict(input_df)
st.subheader("🌟 Predicted Yield")
st.write(f"Estimated Yield: **{prediction[0]:.2f} tons/hectare**")

# Display User Input Summary with the actual selected values
st.subheader("📋 Input Summary")
input_df_display = pd.DataFrame({
    'Region': [input_df.original_region],
    'Soil_Type': [input_df.original_soil],
    'Crop': [input_df.original_crop],
    'Rainfall_mm': input_df['Rainfall_mm'].iloc[0],
    'Temperature_Celsius': input_df['Temperature_Celsius'].iloc[0],
    'Fertilizer_Used': ['Yes' if input_df['Fertilizer_Used'].iloc[0] == 1 else 'No'],
    'Irrigation_Used': ['Yes' if input_df['Irrigation_Used'].iloc[0] == 1 else 'No'],
    'Weather_Condition': [input_df.original_weather],
    'Days_to_Harvest': input_df['Days_to_Harvest'].iloc[0]
})

# Apply highlight_max only to numerical columns
numerical_cols = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
styled_df = input_df_display.style.apply(
    lambda x: ['background-color: yellow' if x.name in numerical_cols and x.max() == v else '' for v in x],
    axis=1
)
st.dataframe(styled_df)

# Prediction Gauge Chart (Using plotly)
st.subheader("🎯 Yield Prediction Gauge")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction[0],
    title={'text': "Tons per Hectare"},
    gauge={
        'axis': {'range': [None, 10]},
        'bar': {'color': "green"},
        'steps': [
            {'range': [0, 3], 'color': "lightgray"},
            {'range': [3, 7], 'color': "yellow"},
            {'range': [7, 10], 'color': "lightgreen"}
        ],
    }
))

st.plotly_chart(fig)

# Rainfall vs. Predicted Yield (Simulation - Adjust for Label Encoding)
import numpy as np
import matplotlib.pyplot as plt

st.subheader("🌧️ Rainfall Impact on Yield (Simulation)")

rainfall_range = np.linspace(0, 1500, 100)
simulated_df = input_df.copy().loc[input_df.index.repeat(100)]
simulated_df['Rainfall_mm'] = rainfall_range
print(simulated_df)

yield_preds = model.predict(simulated_df)

fig, ax = plt.subplots()
ax.plot(rainfall_range, yield_preds, color='skyblue')
ax.set_xlabel("Rainfall (mm)")
ax.set_ylabel("Predicted Yield")
ax.set_title("Predicted Yield vs Rainfall")
st.pyplot(fig)

# Feature Importance (if you used linear models or ensemble)
try:
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
        if hasattr(model, 'coef_'):
            importances = model.coef_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', key=abs, ascending=False)
            st.subheader("🔍 Feature Importances")
            st.bar_chart(feat_df.set_index('Feature'))
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_df = feat_df.sort_values(by='Importance', ascending=False)
            st.subheader("🔍 Feature Importances")
            st.bar_chart(feat_df.set_index('Feature'))
        else:
            st.info("Feature importance not directly available for this model.")
    elif hasattr(model, 'estimators_'): # For VotingRegressor
        # Try to get feature names from the first estimator (assuming consistency)
        first_estimator = model.estimators_[0]
        if hasattr(first_estimator, 'feature_names_in_'):
            feature_names = first_estimator.feature_names_in_
            for name, estimator in model.estimators_:
                if hasattr(estimator, 'coef_'):
                    importances = estimator.coef_
                    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    feat_df = feat_df.sort_values(by='Importance', key=abs, ascending=False)
                    st.subheader(f"🔍 Feature Importances ({name})")
                    st.bar_chart(feat_df.set_index('Feature'))
                    break
                elif hasattr(estimator, 'feature_importances_'):
                    importances = estimator.feature_importances_
                    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    feat_df = feat_df.sort_values(by='Importance', ascending=False)
                    st.subheader(f"🔍 Feature Importances ({name})")
                    st.bar_chart(feat_df.set_index('Feature'))
                    break
            else:
                st.info("Feature importance not available for the individual estimators in the ensemble.")
        else:
            st.info("Feature names not found in the estimators.")
    else:
        st.info("Feature importance information not found in the loaded model.")
except Exception as e:
    st.info(f"Error displaying feature importance: {e}")

# Organize With Tabs or Expanders
tab1, tab2 = st.tabs(["🌟 Prediction", "📊 Visualizations"])

with tab1:
    st.write(f"Estimated Yield: **{prediction[0]:.2f} tons/hectare**")

with tab2:
    st.write("Charts or simulation visuals can be included here.")