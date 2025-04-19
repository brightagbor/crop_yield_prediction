import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lars, BayesianRidge, HuberRegressor
from sklearn.ensemble import VotingRegressor
import joblib

# Load the dataset
data = pd.read_csv("../dataset/crop_yield.csv")

# Separate features (X) and target (y)
X = data.drop(["Yield_tons_per_hectare"], axis=1)
y = data["Yield_tons_per_hectare"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print(X_train.head(20))
print(X_train["Region"].unique())
print(X_train["Soil_Type"].unique())
print(X_train["Crop"].unique())
print(X_train["Weather_Condition"].unique())

# Identify categorical features for Label Encoding
categorical_features = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
boolean_features = ['Fertilizer_Used', 'Irrigation_Used']
numeric_features = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']

# Apply Label Encoding to each categorical feature
label_encoders = {}
for column in categorical_features:
    le = LabelEncoder()
    X_train[column] = le.fit_transform(X_train[column])
    X_test[column] = le.transform(X_test[column])  # Use transform on test data
    label_encoders[column] = le # Optionally store the encoders for later use

# Column transformer to handle remaining (numeric and boolean) features
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numeric_features),
    ('bool', 'passthrough', boolean_features)
], remainder='passthrough')

# Define your base regressors
lr = LinearRegression()
ridge = Ridge()
lar = Lars()
br = BayesianRidge()
huber = HuberRegressor()

# Ensemble regressor
ensemble = VotingRegressor([
    ('lr', lr),
    ('ridge', ridge),
    ('lar', lar),
    ('br', br),
    ('huber', huber)
])

# Create and train the ensemble model
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Ensemble R² with Label Encoding: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save the trained model (the ensemble in this case, as preprocessing is done separately)
joblib.dump(ensemble, 'crop_yield_model_label_encoded.pkl')