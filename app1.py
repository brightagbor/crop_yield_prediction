# import streamlit as st
# import pandas as pd
# import joblib

# # Load pipeline
# model = joblib.load('crop_yield_pipeline.pkl')

# st.title("🌾 Crop Yield Prediction (Auto-Processed)")

# def get_user_input():
#     region = st.sidebar.selectbox("Region", ['North', 'South', 'East', 'West'])
#     soil = st.sidebar.selectbox("Soil Type", ['Loamy', 'Sandy', 'Clay'])
#     crop = st.sidebar.selectbox("Crop", ['Wheat', 'Maize', 'Rice'])
#     weather = st.sidebar.selectbox("Weather Condition", ['Sunny', 'Rainy', 'Cloudy'])
#     rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 1500.0, 500.0)
#     temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
#     fertilizer = st.sidebar.selectbox("Fertilizer Used", ['Yes', 'No'])
#     irrigation = st.sidebar.selectbox("Irrigation Used", ['Yes', 'No'])
#     days_to_harvest = st.sidebar.slider("Days to Harvest", 50, 200, 120)

#     input_data = pd.DataFrame({
#         'Region': [region],
#         'Soil_Type': [soil],
#         'Crop': [crop],
#         'Weather_Condition': [weather],
#         'Rainfall_mm': [rainfall],
#         'Temperature_Celsius': [temperature],
#         'Fertilizer_Used': [1 if fertilizer == 'Yes' else 0],
#         'Irrigation_Used': [1 if irrigation == 'Yes' else 0],
#         'Days_to_Harvest': [days_to_harvest]
#     })

#     return input_data

# input_df = get_user_input()

# print(input_df)


# # Predict
# prediction = model.predict(input_df)
# st.subheader("🌟 Predicted Yield")
# st.write(f"Estimated Yield: **{prediction[0]:.2f} tons/hectare**")


# # Check for missing values in the input DataFrame
# print(input_df.isna().sum())

# # Assuming input_df is your DataFrame

# # Fill NaN values with 0 (or other appropriate handling)
# # input_df = input_df.fillna(0)

# # Now, apply the styling
# # st.dataframe(input_df.style.highlight_max(axis=1))


# # Display User Input Summary
# st.subheader("📋 Input Summary")
# st.dataframe(input_df.style.highlight_max(axis=1))


# # Prediction Gauge Chart (Using plotly)
# import plotly.graph_objects as go

# st.subheader("🎯 Yield Prediction Gauge")

# fig = go.Figure(go.Indicator(
#     mode="gauge+number",
#     value=prediction[0],
#     title={'text': "Tons per Hectare"},
#     gauge={
#         'axis': {'range': [None, 10]},
#         'bar': {'color': "green"},
#         'steps': [
#             {'range': [0, 3], 'color': "lightgray"},
#             {'range': [3, 7], 'color': "yellow"},
#             {'range': [7, 10], 'color': "lightgreen"}
#         ],
#     }
# ))

# st.plotly_chart(fig)

# # Rainfall vs. Predicted Yield
# import numpy as np
# import matplotlib.pyplot as plt

# # st.subheader("🌧️ Rainfall Impact on Yield (Simulation)")

# # rainfall_range = np.linspace(0, 1500, 100)
# # simulated_df = input_df.copy().loc[input_df.index.repeat(100)]
# # simulated_df['Rainfall_mm'] = rainfall_range
# # print(simulated_df)

# # yield_preds = model.predict(simulated_df)

# # fig, ax = plt.subplots()
# # ax.plot(rainfall_range, yield_preds, color='skyblue')
# # ax.set_xlabel("Rainfall (mm)")
# # ax.set_ylabel("Predicted Yield")
# # ax.set_title("Predicted Yield vs Rainfall")
# # st.pyplot(fig)


# # Feature Importance (if you used linear models or ensemble)
# try:
#     importances = model.named_steps['regressor'].estimators_[0].coef_
#     feature_names = model.named_steps['preprocessor'].get_feature_names_out()
#     feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
#     feat_df = feat_df.sort_values(by='Importance', key=abs, ascending=False)

#     st.subheader("🔍 Feature Importances")
#     st.bar_chart(feat_df.set_index('Feature'))
# except:
#     st.info("Feature importance not available for this model setup.")


# # Organize With Tabs or Expanders
# tab1, tab2 = st.tabs(["🌟 Prediction", "📊 Visualizations"])

# with tab1:
#     st.write(f"Estimated Yield: **{prediction[0]:.2f} tons/hectare**")

# with tab2:
#     st.write("Charts or simulation visuals can be included here.")


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the label-encoded trained model
model = joblib.load('crop_yield_model_label_encoded.pkl')

# Define the label mappings based on your training data
region_mapping = {'North': 0, 'South': 1, 'East': 2, 'West': 3}
soil_mapping = {'Loamy': 0, 'Sandy': 1, 'Clay': 2}
crop_mapping = {'Wheat': 0, 'Maize': 1, 'Rice': 2}
weather_mapping = {'Sunny': 0, 'Rainy': 1, 'Cloudy': 2}

st.title("🌾 Crop Yield Prediction (Label Encoded)")

def get_user_input():
    region = st.sidebar.selectbox("Region", list(region_mapping.keys()))
    soil = st.sidebar.selectbox("Soil Type", list(soil_mapping.keys()))
    crop = st.sidebar.selectbox("Crop", list(crop_mapping.keys()))
    weather = st.sidebar.selectbox("Weather Condition", list(weather_mapping.keys()))
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 1500.0, 500.0)
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
    fertilizer = st.sidebar.selectbox("Fertilizer Used", ['Yes', 'No'])
    irrigation = st.sidebar.selectbox("Irrigation Used", ['Yes', 'No'])
    days_to_harvest = st.sidebar.slider("Days to Harvest", 50, 200, 120)

    input_data = pd.DataFrame({
        'Region': [region_mapping[region]],
        'Soil_Type': [soil_mapping[soil]],
        'Crop': [crop_mapping[crop]],
        'Rainfall_mm': [rainfall],
        'Temperature_Celsius': [temperature],
        'Fertilizer_Used': [1 if fertilizer == 'Yes' else 0],
        'Irrigation_Used': [1 if irrigation == 'Yes' else 0],
        'Weather_Condition': [weather_mapping[weather]],
        'Days_to_Harvest': [days_to_harvest]
    })

    # Ensure the column order matches the training data after label encoding
    input_data = input_data[['Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
                             'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition', 'Days_to_Harvest']]

    return input_data

input_df = get_user_input()

print(input_df)

# Predict
prediction = model.predict(input_df)
st.subheader("🌟 Predicted Yield")
st.write(f"Estimated Yield: **{prediction[0]:.2f} tons/hectare**")

# Display User Input Summary with original labels
st.subheader("📋 Input Summary")
input_df_display = input_df.copy()
input_df_display.replace({v: k for k, v in region_mapping.items()}, inplace=True)
input_df_display.replace({v: k for k, v in soil_mapping.items()}, inplace=True)
input_df_display.replace({v: k for k, v in crop_mapping.items()}, inplace=True)
input_df_display.replace({v: k for k, v in weather_mapping.items()}, inplace=True)

# st.dataframe(input_df_display.style.highlight_max(axis=1))

# Prediction Gauge Chart (Using plotly)
import plotly.graph_objects as go

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

# st.subheader("🌧️ Rainfall Impact on Yield (Simulation)")

# rainfall_range = np.linspace(0, 1500, 100)
# simulated_df = input_df.copy().loc[input_df.index.repeat(100)]
# simulated_df['Rainfall_mm'] = rainfall_range
# print(simulated_df)

# yield_preds = model.predict(simulated_df)

# fig, ax = plt.subplots()
# ax.plot(rainfall_range, yield_preds, color='skyblue')
# ax.set_xlabel("Rainfall (mm)")
# ax.set_ylabel("Predicted Yield")
# ax.set_title("Predicted Yield vs Rainfall")
# st.pyplot(fig)

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