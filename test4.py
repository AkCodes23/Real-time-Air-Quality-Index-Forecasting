# smote is not working for continuous AQI values , need it for classification

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

# Load the dataset
data_path = r'D:\Projects\AQI\Final records\finetuning\processed_city_day.csv'
data = pd.read_csv(data_path)
print("Original Data Shape:", data.shape)
print("First few rows of the data:\n", data.head())
print("Columns in DataFrame:", data.columns)

# Ensure the correct columns are present
required_cols = ['city', 'date', 'PM2.5', 'NO2', 'SO2', 'AQI']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"Missing columns in the dataset: {missing_cols}")
else:
    # Select relevant columns and replace missing values with mean
    data = data[required_cols].copy()

    # Replace missing values with mean for selected columns
    imputer = SimpleImputer(strategy='mean')
    data[['PM2.5', 'NO2', 'SO2', 'AQI']] = imputer.fit_transform(data[['PM2.5', 'NO2', 'SO2', 'AQI']])

    # Save processed dataset
    processed_data_path = r'd:/Projects/AQI/Final records/processed_city_day.csv'
    data.to_csv(processed_data_path, index=False)

    # Prepare data for training (X) and target variable (y)
    X = data[['PM2.5', 'NO2', 'SO2']]
    y = data['AQI'] 

    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Split the data into training and testing sets with different ratios
    ratios = [0.2, 0.3, 0.4]
    split_data = {}

    for ratio in ratios:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=42)
        X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=ratio, random_state=42)
        split_data[ratio] = {
            'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
            'X_train_smote': X_train_smote, 'X_test_smote': X_test_smote, 'y_train_smote': y_train_smote, 'y_test_smote': y_test_smote
        }

    # Standardize features
    scaler = StandardScaler()

    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'KNN': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor()
    }

    # Function to train and save models
    def train_and_save_models(models, X_train, y_train, suffix=''):
        for name, model in models.items():
            model.fit(X_train, y_train)
            joblib.dump(model, f'd:/Projects/AQI/Final records/{name.lower().replace(" ", "_")}_model{suffix}.pkl')

    # Function to calculate metrics
    def calculate_metrics(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, rmse, mae, r2

    metrics = {}
    for ratio, data_split in split_data.items():
        X_train_scaled = scaler.fit_transform(data_split['X_train'])
        X_test_scaled = scaler.transform(data_split['X_test'])
        X_train_smote_scaled = scaler.fit_transform(data_split['X_train_smote'])
        X_test_smote_scaled = scaler.transform(data_split['X_test_smote'])

        suffix = f'_ratio_{int(ratio * 100)}'
        train_and_save_models(models, X_train_scaled, data_split['y_train'], suffix)
        train_and_save_models(models, X_train_smote_scaled, data_split['y_train_smote'], f'_smote{suffix}')

        for name, model in models.items():
            model_metrics = calculate_metrics(model, X_test_scaled, data_split['y_test'])
            metrics[f'{name}{suffix}'] = model_metrics

            model_metrics_smote = calculate_metrics(model, X_test_smote_scaled, data_split['y_test_smote'])
            metrics[f'{name}_smote{suffix}'] = model_metrics_smote

    # Print accuracy metrics for each model
    print("\nAccuracy Metrics:")
    for name, (mse, rmse, mae, r2) in metrics.items():
        print(f"\n{name}:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R-squared: {r2:.2f}")

    # Plot feature importance scores
    best_model_name = max(metrics, key=lambda name: metrics[name][3])  # R-squared is at index 3
    best_model = models[best_model_name.split('_')[0]]
    importances = best_model.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=feature_names)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Feature Importance Scores')
    plt.show()

    # Generate predictions for multiple cities and plot results
    cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']

    def future_aqi_ml(model, city, days=3650):
        city_data = data[data['city'] == city].copy()
        city_data['date'] = pd.to_datetime(city_data['date'], format='%d-%m-%Y')
        city_data.set_index('date', inplace=True)

        # Generate future dates
        future_dates = pd.date_range(start=city_data.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')

        # Predict future pollutants
        future_pm25 = np.array(city_data['PM2.5'].rolling(window=30, min_periods=1).mean().iloc[-1] + np.random.normal(0, 0.1, days))
        future_no2 = np.array(city_data['NO2'].rolling(window=30, min_periods=1).mean().iloc[-1] + np.random.normal(0, 0.1, days))
        future_so2 = np.array(city_data['SO2'].rolling(window=30, min_periods=1).mean().iloc[-1] + np.random.normal(0, 0.1, days))

        future_X = pd.DataFrame({
            'PM2.5': future_pm25,
            'NO2': future_no2,
            'SO2': future_so2
        })

        future_X_scaled = scaler.transform(future_X)
        future_aqi = model.predict(future_X_scaled)

        # Create a DataFrame with future predictions
        future_aqi_df = pd.DataFrame({
            'Date': future_dates,
            'AQI': future_aqi
        })

        return future_aqi_df

    for city in cities:
        future_aqi_df_ml = future_aqi_ml(best_model, city, days=3650)
        print(f"\nFuture AQI Predictions for {city} for the next 10 years using {best_model_name}:\n", future_aqi_df_ml.head())

        # Plot results
        plt.figure(figsize=(12, 6))
        daily_avg_aqi = data[data['city'] == city].copy()
        daily_avg_aqi['date'] = pd.to_datetime(daily_avg_aqi['date'], format='%d-%m-%Y')
        daily_avg_aqi.set_index('date', inplace=True)
        daily_avg_aqi = daily_avg_aqi.drop(columns=['city']).resample('D').mean().reset_index()
        sns.lineplot(data=daily_avg_aqi, x='date', y='AQI', label='Historical AQI')
        sns.lineplot(data=future_aqi_df_ml, x='Date', y='AQI', label=f'{best_model_name} Future AQI')
        plt.xlabel('Year')
        plt.ylabel('AQI')
        plt.title(f'{city} AQI Forecasting')
        plt.legend()
        plt.show()

    # Plot Line Charts for Each Year Before and After Data Normalization
    for city in cities:
        for year in range(2015, 2025):
            yearly_data = data[(data['city'] == city) & (data['date'].str.contains(str(year)))].copy()

            plt.figure(figsize=(12, 6))
            sns.lineplot(data=yearly_data, x='date', y='AQI', label='Original AQI')
            sns.lineplot(data=yearly_data, x='date', y='AQI', label='Normalized AQI')
            plt.xlabel('Date')
            plt.ylabel('AQI')
            plt.title(f'{city} AQI Line Chart for {year}')
            plt.legend()
            plt.show()

    # True vs Predicted values plot for each model
    for name, model in models.items():
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, model.predict(X_test_scaled), alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('True AQI')
        plt.ylabel('Predicted AQI')
        plt.title(f'{name} True vs Predicted AQI')
        plt.show()

    # COVID Period Analysis
    for city in cities:
        covid_period = data[(data['city'] == city) & (data['date'].str.contains('2019|2020|2021|2022'))].copy()
        covid_period['year_month'] = covid_period['date'].str.slice(0, 7)
        avg_covid_aqi = covid_period.groupby('year_month')['AQI'].mean()

        plt.figure(figsize=(12, 6))
        sns.lineplot(data=avg_covid_aqi)
        plt.xlabel('Year-Month')
        plt.ylabel('Average AQI')
        plt.title(f'{city} Average AQI During COVID Period (2019-2022)')
        plt.show()

    # Additional outputs for SMOTE and non-SMOTE methods
    for ratio, data_split in split_data.items():
        for name, model in models.items():
            y_pred = model.predict(scaler.transform(data_split['X_test']))
            y_pred_smote = model.predict(scaler.transform(data_split['X_test_smote']))

            print(f"\n{name} Results for test size {ratio}:")
            print(f"Non-SMOTE - MSE: {mean_squared_error(data_split['y_test'], y_pred):.2f}, MAE: {mean_absolute_error(data_split['y_test'], y_pred):.2f}, R-squared: {r2_score(data_split['y_test'], y_pred):.2f}")
            print(f"SMOTE - MSE: {mean_squared_error(data_split['y_test_smote'], y_pred_smote):.2f}, MAE: {mean_absolute_error(data_split['y_test_smote'], y_pred_smote):.2f}, R-squared: {r2_score(data_split['y_test_smote'], y_pred_smote):.2f}")
