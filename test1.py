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
from statsmodels.tsa.arima.model import ARIMA
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import warnings
warnings.filterwarnings("ignore")

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

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE oversampling
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_scaled, y)

    def train_and_evaluate(X_train, y_train, X_test, y_test, smote_applied=False):
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
            model.fit(X_train, y_train)
            suffix = "_smote" if smote_applied else ""
            joblib.dump(model, f'd:/Projects/AQI/Final records/{name.lower().replace(" ", "_")}{suffix}_model.pkl')

        # Accuracy Metrics Calculation
        def calculate_metrics(model, X_test, y_test):
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            return mse, rmse, mae, r2

        metrics = {name: calculate_metrics(model, X_test, y_test) for name, model in models.items()}

        # Identify the most accurate model based on R-squared value
        best_model_name = max(metrics, key=lambda name: metrics[name][3])  # R-squared is at index 3
        best_model = models[best_model_name]
        print(f"\nMost accurate model {'with SMOTE' if smote_applied else 'without SMOTE'}: {best_model_name}")

        # Print accuracy metrics for each model
        print("\nAccuracy Metrics:")
        for name, (mse, rmse, mae, r2) in metrics.items():
            print(f"\n{name}:")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R-squared: {r2:.2f}")

        return best_model, metrics

    # Split the data into training and testing sets
    def train_test_split_and_evaluate(X, y, smote_applied=False):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return train_and_evaluate(X_train, y_train, X_test, y_test, smote_applied)

    # Evaluate without SMOTE
    best_model_no_smote, metrics_no_smote = train_test_split_and_evaluate(X_scaled, y, smote_applied=False)

    # Evaluate with SMOTE
    best_model_smote, metrics_smote = train_test_split_and_evaluate(X_smote, y_smote, smote_applied=True)

    # ARIMA model for time series forecasting
    def arima_model(data, order=(5, 1, 0)):
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        model = ARIMA(data['AQI'], order=order)
        arima_fit = model.fit()
        forecast = arima_fit.forecast(steps=30)  # 30 days into the future
        return arima_fit, forecast

    arima_fit, arima_forecast = arima_model(data)

    # Plotting Functions
    def plot_true_vs_predicted(model, X_train, y_train, X_test, y_test, smote_applied=False):
        plt.figure(figsize=(12, 6))
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        plt.plot(y_train.values, label='True Train Values')
        plt.plot(y_pred_train, label='Predicted Train Values')
        plt.title(f'Comparison of True vs Predicted Train Values {"with SMOTE" if smote_applied else "without SMOTE"}')
        plt.xlabel('Sample Index')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='True Test Values')
        plt.plot(y_pred_test, label='Predicted Test Values')
        plt.title(f'Comparison of True vs Predicted Test Values {"with SMOTE" if smote_applied else "without SMOTE"}')
        plt.xlabel('Sample Index')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Scatter plot of true vs predicted AQI
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred_test, alpha=0.5)
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
        plt.title(f'Scatter Plot of True vs Predicted AQI {"with SMOTE" if smote_applied else "without SMOTE"}')
        plt.xlabel('True AQI')
        plt.ylabel('Predicted AQI')
        plt.grid(True)
        plt.show()

    def plot_feature_importance(model, X, smote_applied=False):
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
            features = X.columns if isinstance(X, pd.DataFrame) else ['PM2.5', 'NO2', 'SO2']
            plt.figure(figsize=(8, 6))
            sns.barplot(x=feature_importances, y=features)
            plt.title(f'Feature Importance {"with SMOTE" if smote_applied else "without SMOTE"}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.grid(True)
            plt.show()

    # Plot ARIMA predictions
    def plot_arima_predictions(arima_fit, forecast):
        plt.figure(figsize=(12, 6))
        plt.plot(arima_fit.fittedvalues, label='ARIMA Fitted Values')
        plt.plot(forecast, label='ARIMA Forecast')
        plt.title('ARIMA Model Fitted Values and Forecast')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot true vs predicted for best models
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    plot_true_vs_predicted(best_model_no_smote, X_train, y_train, X_test, y_test, smote_applied=False)
    plot_feature_importance(best_model_no_smote, X, smote_applied=False)

    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)
    plot_true_vs_predicted(best_model_smote, X_train_smote, y_train_smote, X_test_smote, y_test_smote, smote_applied=True)
    plot_feature_importance(best_model_smote, X, smote_applied=True)

    plot_arima_predictions(arima_fit, arima_forecast)

    # Additional Data Visualization
    def plot_general_data_analysis(data):
        # Raw data over time
        data.set_index('date', inplace=True)
        plt.figure(figsize=(14, 7))
        plt.plot(data['AQI'], label='AQI')
        plt.title('AQI Over Time')
        plt.xlabel('Date')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Histogram
        data.hist(figsize=(14, 7), bins=20)
        plt.show()

        # Correlation heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

        # Line chart for AQI over years
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        data.resample('Y')['AQI'].mean().plot(kind='line', figsize=(14, 7), title='Yearly Average AQI')
        plt.show()

    plot_general_data_analysis(data)
