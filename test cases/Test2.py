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

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train and save models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(),
        'AdaBoost': AdaBoostRegressor(),
        'KNN': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor()
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, f'd:/Projects/AQI/Final records/{name.lower().replace(" ", "_")}_model.pkl')

    # Accuracy Metrics Calculation
    def calculate_metrics(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, rmse, mae, r2

    metrics = {name: calculate_metrics(model, X_test_scaled, y_test) for name, model in models.items()}

    # Identify the most accurate model based on R-squared value
    best_model_name = max(metrics, key=lambda name: metrics[name][3])  # R-squared is at index 3
    best_model = models[best_model_name]
    print(f"\nMost accurate model: {best_model_name}")

    # Print accuracy metrics for each model
    print("\nAccuracy Metrics:")
    for name, (mse, rmse, mae, r2) in metrics.items():
        print(f"\n{name}:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R-squared: {r2:.2f}")

    # Future AQI prediction using the most accurate machine learning model
    def future_aqi_ml(model, city, days=3650):
        city_data = data[data['city'] == city].copy()
        city_data['date'] = pd.to_datetime(city_data['date'], format='%d-%m-%Y')
        city_data.set_index('date', inplace=True)
        
        # Generate future dates
        future_dates = pd.date_range(start=city_data.index[-1] + pd.Timedelta(days=1), periods=days, freq='D')
        
        # Predict future pollutants (simple assumption: using a moving average or a slight increase/decrease)
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

    future_aqi_df_ml = future_aqi_ml(best_model, 'Ahmedabad', days=3650)
    print(f"\nFuture AQI Predictions for Ahmedabad for the next 10 years using {best_model_name}:\n", future_aqi_df_ml.head())

    # Plot AQI parameter comparison for a city on a daily basis using the best ML model
    plt.figure(figsize=(12, 6))
    daily_avg_aqi = data[data['city'] == 'Ahmedabad'].copy()
    daily_avg_aqi['date'] = pd.to_datetime(daily_avg_aqi['date'], format='%d-%m-%Y')
    daily_avg_aqi.set_index('date', inplace=True)
    daily_avg_aqi = daily_avg_aqi.drop(columns=['city']).resample('D').mean().reset_index()
    sns.lineplot(data=daily_avg_aqi, x='date', y='AQI', label='Historical AQI')
    sns.lineplot(data=future_aqi_df_ml, x='Date', y='AQI', label=f'Future AQI ({best_model_name})', linestyle='-.')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title(f'Daily AQI for Ahmedabad with Forecast till 2030')
    plt.grid(True) 
    plt.legend()
    plt.show()

    # Comparison of daily average AQI between cities
    daily_avg_aqi_all_cities = data.copy()
    daily_avg_aqi_all_cities['date'] = pd.to_datetime(daily_avg_aqi_all_cities['date'], format='%d-%m-%Y')
    daily_avg_aqi_all_cities.set_index('date', inplace=True)
    daily_avg_aqi_all_cities = daily_avg_aqi_all_cities.groupby(['city']).resample('D').mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_avg_aqi_all_cities, x='date', y='AQI', hue='city')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('Daily Average AQI for Different Cities')
    plt.grid(True) 
    plt.legend()
    plt.show()

    # Plot the contribution of different pollutants over time for a city
    pollutants = ['PM2.5', 'NO2', 'SO2']
    plt.figure(figsize=(12, 6))
    for pollutant in pollutants:
        sns.lineplot(data=daily_avg_aqi, x='date', y=pollutant, label=pollutant)
    plt.xlabel('Date')
    plt.ylabel('Pollutant Level')
    plt.title(f'Pollutant Levels for Ahmedabad Over Time')
    plt.grid(True) 
    plt.legend()
    plt.show()

    # Additional plots from the provided PDF
    # Example: Scatter plot of PM2.5 vs AQI
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='PM2.5', y='AQI')
    plt.xlabel('PM2.5')
    plt.ylabel('AQI')
    plt.title('Scatter plot of PM2.5 vs AQI')
    plt.grid(True)
    plt.show()

    # Example: Heatmap of correlation between pollutants and AQI
    plt.figure(figsize=(8, 6))
    correlation = data[['PM2.5', 'NO2', 'SO2', 'AQI']].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Heatmap of Correlation between Pollutants and AQI')
    plt.show()

    # Example: Box plot of AQI by city
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='city', y='AQI')
    plt.xlabel('City')
    plt.ylabel('AQI')
    plt.title('Box plot of AQI by City')
    plt.xticks(rotation=45)
    plt.show()

    # Example: Time series plot of AQI over time for Ahmedabad
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=daily_avg_aqi, x='date', y='AQI')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('Time Series of AQI for Ahmedabad')
    plt.grid(True)
    plt.show()

    # Example: Pair plot of pollutants and AQI
    sns.pairplot(data[['PM2.5', 'NO2', 'SO2', 'AQI']])
    plt.suptitle('Pair Plot of Pollutants and AQI')
    plt.show()

    # Example: Distribution plot of pollutants and AQI
    plt.figure(figsize=(8, 6))
    sns.histplot(data['PM2.5'], kde=True, color='blue', label='PM2.5')
    sns.histplot(data['NO2'], kde=True, color='red', label='NO2')
    sns.histplot(data['SO2'], kde=True, color='green', label='SO2')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of Pollutants')
    plt.legend()
    plt.show()

    # Example: Violin plot of pollutants by city
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=data, x='city', y='PM2.5')
    plt.xlabel('City')
    plt.ylabel('PM2.5')
    plt.title('Violin Plot of PM2.5 by City')
    plt.xticks(rotation=45)
    plt.show()
