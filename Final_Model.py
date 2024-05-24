import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the dataset
data_path = r'D:\Projects\AQI\Datasets\Final_Dataset.csv'
data = pd.read_csv(data_path, sep=',', quotechar='"', decimal=',')
print("Original Data Shape:", data.shape)

# Print the first few rows to understand the structure
print("First few rows of the data:\n", data.head())

# Print the column names
print("Columns in DataFrame:", data.columns)

# Ensure the correct columns are present
required_cols = ['city', 'date', 'PM2.5', 'NO2', 'SO2', 'AQI']
missing_cols = [col for col in required_cols if col not in data.columns]
if missing_cols:
    print(f"Missing columns in the dataset: {missing_cols}")
else:
    # Select relevant columns and replace missing values with mean
    data = data[required_cols].copy()

    # Replace missing values with mean for selected columns
    imputer = SimpleImputer(strategy='mean')
    data[['PM2.5', 'NO2', 'SO2', 'AQI']] = imputer.fit_transform(data[['PM2.5', 'NO2', 'SO2', 'AQI']])

    # Prepare data for training (X) and target variable (y)
    X = data[['PM2.5', 'NO2', 'SO2']]  # Features
    y = data['AQI']  # Target variable (Air Quality Index)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    joblib.dump(model_rf, 'random_forest_model.pkl')

    # Train Linear Regression model
    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    joblib.dump(model_lr, 'linear_regression_model.pkl')

    # Train Support Vector Regression (SVR) model
    model_svr = SVR()
    model_svr.fit(X_train_scaled, y_train)
    joblib.dump(model_svr, 'svr_model.pkl')

    # Train XGBoost model
    model_xgb = XGBRegressor()
    model_xgb.fit(X_train_scaled, y_train)
    joblib.dump(model_xgb, 'xgboost_model.pkl')

    # Train AdaBoost model
    model_ada = AdaBoostRegressor()
    model_ada.fit(X_train_scaled, y_train)
    joblib.dump(model_ada, 'adaboost_model.pkl')

    # Train KNN model
    model_knn = KNeighborsRegressor()
    model_knn.fit(X_train_scaled, y_train)
    joblib.dump(model_knn, 'knn_model.pkl')

    # Function to predict AQI for a specific date and location
    def predict_aqi(city, date, pm25, no2, so2):
        input_data = pd.DataFrame([[pm25, no2, so2]],
                                  columns=['PM2.5', 'NO2', 'SO2'])
        input_data_imputed = imputer.transform(input_data)
        input_scaled = scaler.transform(input_data_imputed)

        model_rf = joblib.load('random_forest_model.pkl')
        aqi_pred_rf = model_rf.predict(input_scaled)

        model_lr = joblib.load('linear_regression_model.pkl')
        aqi_pred_lr = model_lr.predict(input_scaled)

        model_svr = joblib.load('svr_model.pkl')
        aqi_pred_svr = model_svr.predict(input_scaled)

        model_xgb = joblib.load('xgboost_model.pkl')
        aqi_pred_xgb = model_xgb.predict(input_scaled)

        model_ada = joblib.load('adaboost_model.pkl')
        aqi_pred_ada = model_ada.predict(input_scaled)

        model_knn = joblib.load('knn_model.pkl')
        aqi_pred_knn = model_knn.predict(input_scaled)

        # Combine predictions into a DataFrame for easier plotting
        predictions = pd.DataFrame({
            'Model': ['Random Forest', 'Linear Regression', 'SVR', 'XGBoost', 'AdaBoost', 'KNN'],
            'AQI': [aqi_pred_rf[0], aqi_pred_lr[0], aqi_pred_svr[0], aqi_pred_xgb[0], aqi_pred_ada[0], aqi_pred_knn[0]]
        })

        return predictions

    # Example usage of the predict_aqi function
    print("\nAQI Predictions for a given date and location:")
    city = input("Enter the city: ")
    date = input("Enter the date (DD-MM-YYYY): ")
    pm25 = float(input("Enter the PM2.5 value: "))
    no2 = float(input("Enter the NO2 value: "))
    so2 = float(input("Enter the SO2 value: "))

    predictions = predict_aqi(city, date, pm25, no2, so2)

    for index, row in predictions.iterrows():
        print(f"{row['Model']} Prediction: {row['AQI']:.2f}")

    # Calculate accuracy metrics for each model
    y_pred_rf = model_rf.predict(X_test_scaled)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    y_pred_lr = model_lr.predict(X_test_scaled)
    mse_lr = mean_squared_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    y_pred_svr = model_svr.predict(X_test_scaled)
    mse_svr = mean_squared_error(y_test, y_pred_svr)
    r2_svr = r2_score(y_test, y_pred_svr)

    y_pred_xgb = model_xgb.predict(X_test_scaled)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb = r2_score(y_test, y_pred_xgb)

    y_pred_ada = model_ada.predict(X_test_scaled)
    mse_ada = mean_squared_error(y_test, y_pred_ada)
    r2_ada = r2_score(y_test, y_pred_ada)

    y_pred_knn = model_knn.predict(X_test_scaled)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)

    # Print accuracy metrics for each model
    print("\nAccuracy Metrics:")
    print("Random Forest:")
    print(f"MSE: {mse_rf:.2f}")
    print(f"R-squared: {r2_rf:.2f}")

    print("\nLinear Regression:")
    print(f"MSE: {mse_lr:.2f}")
    print(f"R-squared: {r2_lr:.2f}")

    print("\nSupport Vector Regression (SVR):")
    print(f"MSE: {mse_svr:.2f}")
    print(f"R-squared: {r2_svr:.2f}")

    print("\nXGBoost:")
    print(f"MSE: {mse_xgb:.2f}")
    print(f"R-squared: {r2_xgb:.2f}")

    print("\nAdaBoost:")
    print(f"MSE: {mse_ada:.2f}")
    print(f"R-squared: {r2_ada:.2f}")

    print("\nK-Nearest Neighbors (KNN):")
    print(f"MSE: {mse_knn:.2f}")
    print(f"R-squared: {r2_knn:.2f}")

    # Visualizations

    # Scatter plots for each feature vs AQI
    for feature in ['PM2.5', 'NO2', 'SO2']:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=data[feature], y=data['AQI'])
        plt.xlabel(feature)
        plt.ylabel('AQI')
        plt.title(f'{feature} vs AQI')
        plt.grid(True)
        plt.show()

    # Residual plots for each model
    for model, name in [(model_rf, 'Random Forest'), (model_lr, 'Linear Regression'), (model_svr, 'SVR'),
                        (model_xgb, 'XGBoost'), (model_ada, 'AdaBoost'), (model_knn, 'KNN')]:
        y_pred = model.predict(X_test_scaled)
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted AQI')
        plt.ylabel('Residuals')
        plt.title(f'Residuals for {name}')
        plt.grid(True)
        plt.show()

    # Histograms of features and AQI
    for feature in ['PM2.5', 'NO2', 'SO2', 'AQI']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], bins=30, kde=True)
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {feature}')
        plt.grid(True)
        plt.show()

    # Histograms of features with respect to time period (monthly)
    data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
    data['Month'] = data['date'].dt.month
    for feature in ['PM2.5', 'NO2', 'SO2']:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, x='Month', y=feature, bins=12, kde=True)
        plt.xlabel('Month')
        plt.ylabel(feature)
        plt.title(f'Histogram of {feature} by Month')
        plt.grid(True)
        plt.show()
