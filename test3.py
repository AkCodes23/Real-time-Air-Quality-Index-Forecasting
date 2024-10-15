import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import random

# Load the dataset
data = pd.read_csv('air_quality.csv')  # Replace with your dataset
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')

# Feature Selection and Preprocessing
X = data[['PM2.5', 'NO2', 'SO2']]  # Replace with your features
y = data['AQI']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
models = {
    'Random Forest': RandomForestRegressor(),
    'XGBoost': XGBRegressor(),
    'AdaBoost': AdaBoostRegressor()
}

metrics = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test)
    r2 = r2_score(y_test, y_pred)
    
    metrics[name] = (mse, rmse, mae, r2)

# Selecting the Best Model
best_model = models['Random Forest']  # Choose based on evaluation

# Function to Predict Future AQI
def future_aqi_ml(model, city, days=3650):
    city_data = data[data['city'] == city].copy()
    future_dates = pd.date_range(city_data['date'].max(), periods=days, freq='D')
    future_data = pd.DataFrame(index=future_dates)
    future_data['PM2.5'] = city_data['PM2.5'].mean()  # Replace with relevant values or predictions
    future_data['NO2'] = city_data['NO2'].mean()
    future_data['SO2'] = city_data['SO2'].mean()

    future_data_scaled = scaler.transform(future_data)
    future_data['AQI'] = model.predict(future_data_scaled)
    
    return future_data

# Predict for Multiple Cities
cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
future_predictions = {}

for city in cities:
    future_predictions[city] = future_aqi_ml(best_model, city, days=3650)
    print(f"\nFuture AQI Predictions for {city}:\n", future_predictions[city].head())

# Line Charts for Each Year (2015-2024) Before and After Normalization
for city in cities:
    plt.figure(figsize=(12, 6))
    city_data = data[data['city'] == city].copy()
    city_data['date'] = pd.to_datetime(city_data['date'], format='%d-%m-%Y')
    city_data.set_index('date', inplace=True)
    sns.lineplot(data=city_data.resample('Y').mean(), x=city_data.index.year, y='AQI', label=f'{city} AQI (Before Normalization)')
    
    city_data_scaled = pd.DataFrame(scaler.fit_transform(city_data[['PM2.5', 'NO2', 'SO2', 'AQI']]), columns=['PM2.5', 'NO2', 'SO2', 'AQI'], index=city_data.index)
    sns.lineplot(data=city_data_scaled.resample('Y').mean(), x=city_data.index.year, y='AQI', label=f'{city} AQI (After Normalization)', linestyle='--')
    
    plt.xlabel('Year')
    plt.ylabel('AQI')
    plt.title(f'AQI Trend for {city} (2015-2024)')
    plt.legend()
    plt.show()

# Feature Importance Score (Bar Graph)
if hasattr(best_model, 'feature_importances_'):
    feature_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_importances.sort_values().plot(kind='barh', figsize=(8, 6))
    plt.title('Feature Importance Scores')
    plt.show()

# Sample Output for a Random Year and Specific Date
sample_year = random.choice(range(2015, 2025))
sample_date = f'{sample_year}-{random.choice(range(1, 13))}-{random.choice(range(1, 29))}'
sample_city = 'Ahmedabad'
print(f"\nSample Output for {sample_city} on {sample_date}:")

sample_features = np.array([50, 20, 10]).reshape(1, -1)  # Example pollutant levels
sample_features_scaled = scaler.transform(sample_features)
predicted_aqi = best_model.predict(sample_features_scaled)
print(f"AQI Prediction: {predicted_aqi[0]:.2f}")

# Plot of True vs. Predicted Values for Each Model
for name, model in models.items():
    y_pred = model.predict(X_test_scaled)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel('True AQI')
    plt.ylabel('Predicted AQI')
    plt.title(f'True vs. Predicted AQI for {name}')
    plt.grid(True)
    plt.show()

# Model Performance Metrics
print("\nModel Performance Metrics:")
for name, (mse, rmse, mae, r2) in metrics.items():
    print(f"\n{name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")

# Experiment with Different Train-Test Ratios
ratios = [0.7, 0.8, 0.9]
for ratio in ratios:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-ratio), random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test)
        r2 = r2_score(y_test, y_pred)
        print(f"\n{ratio*100}% Train - {name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")

# COVID Analysis (2019-2022)
covid_data = data[(data['date'] >= '2019-01-01') & (data['date'] <= '2022-12-31')].copy()
covid_data['month'] = covid_data['date'].dt.to_period('M')
monthly_avg_aqi = covid_data.groupby(['month', 'city'])['AQI'].mean().unstack()

plt.figure(figsize=(12, 6))
monthly_avg_aqi.plot()
plt.title('Monthly Average AQI During COVID (2019-2022)')
plt.ylabel('Average AQI')
plt.grid(True)
plt.show()

# SMOTE vs Non-SMOTE Analysis
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

for name, model in models.items():
    model.fit(X_resampled, y_resampled)
    y_pred_resampled = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred_resampled)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test)
    r2 = r2_score(y_test, y_pred_resampled)
    print(f"\n{name} with SMOTE - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
