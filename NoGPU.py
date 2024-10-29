import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import statsmodels.api as sm

# Load the dataset
data_path = r"D:\Projects\AQI\Final records\processed_city_day.csv"
data = pd.read_csv(data_path)
print("Original Data Shape:", data.shape)
print("First few rows of the data:\n", data.head())
print("Columns in DataFrame:", data.columns)

# Plot raw data
plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='date', y='AQI')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('Raw AQI Data')
plt.grid(True)
plt.show()

# Plot histograms
data.hist(bins=30, figsize=(20, 15))
plt.suptitle('Histograms of Features')
plt.show()

# Plot heatmap of correlations
plt.figure(figsize=(12, 6))
numeric_data = data.select_dtypes(include=[np.number])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap of Feature Correlations')
plt.show()

# Convert the 'date' column to datetime if not already done
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)

# Filter data for the years 2015-2024
data_filtered = data[(data.index.year >= 2015) & (data.index.year <= 2024)]

# Plot line charts for each year
plt.figure(figsize=(12, 6))
for year in range(2015, 2025):
    yearly_data = data_filtered[data_filtered.index.year == year]
    yearly_data = yearly_data.reset_index()
    sns.lineplot(data=yearly_data, x='date', y='AQI', label=str(year))
plt.xlabel('Date')
plt.ylabel('AQI')
plt.title('Raw AQI Data (2015-2024)')
plt.legend()
plt.grid(True)
plt.show()

# Normalize the AQI data
scaler_aqi = StandardScaler()
data_filtered['AQI_normalized'] = scaler_aqi.fit_transform(data_filtered[['AQI']])

# Plot normalized AQI data for each year
plt.figure(figsize=(12, 6))
for year in range(2015, 2025):
    yearly_data = data_filtered[data_filtered.index.year == year].reset_index()
    sns.lineplot(data=yearly_data, x='date', y='AQI_normalized', label=str(year))
plt.xlabel('Date')
plt.ylabel('Normalized AQI')
plt.title('Normalized AQI Data (2015-2024)')
plt.legend()
plt.grid(True)
plt.show()


# Print all columns in the dataset
print("All columns in the dataset:", data.columns.tolist())
# Ensure the correct columns are present
required_cols = ['city', 'date', 'PM2.5', 'NO2', 'SO2', 'AQI']
data_reset = data.reset_index()  # Reset index to include 'date' in columns
missing_cols = [col for col in required_cols if col not in data_reset.columns]
if missing_cols:
    raise ValueError(f"Missing columns in the dataset: {missing_cols}")

# Select relevant columns and replace missing values with mean
data = data_reset[required_cols].copy()
imputer = SimpleImputer(strategy='mean')
data[['PM2.5', 'NO2', 'SO2', 'AQI']] = imputer.fit_transform(data[['PM2.5', 'NO2', 'SO2', 'AQI']])

# Convert the 'date' column to datetime
data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y')
data.set_index('date', inplace=True)



def handle_outliers_z_score(df, columns, threshold=3):
    for column in columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = np.where(np.abs((df[column] - mean) / std) > threshold, mean, df[column])
    return df

data = handle_outliers_z_score(data, ['PM2.5', 'NO2', 'SO2', 'AQI'], threshold=3)

# Prepare data for training (X) and target variable (y)
X = data[['PM2.5', 'NO2', 'SO2']]
y = data['AQI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

X_train_tensor_lstm = X_train_tensor.unsqueeze(1)  # seq_len is set to 1
X_test_tensor_lstm = X_test_tensor.unsqueeze(1)  # seq_len is set to 1

class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

class GRU(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])
        return out

# Train and evaluate models
models = {
    'LSTM': LSTM(),
    'GRU': GRU(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Linear Regression': LinearRegression(),
    'SVR': SVR(),
    'XGBoost': XGBRegressor(),
    'AdaBoost': AdaBoostRegressor(),
    'KNN': KNeighborsRegressor(),
    'Decision Tree': DecisionTreeRegressor(),
    'SARIMA': sm.tsa.statespace.SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
}

# Scale the target variable y
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Convert scaled y values to tensors
y_train_tensor_scaled = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1)
y_test_tensor_scaled = torch.tensor(y_test_scaled, dtype=torch.float32).view(-1, 1)

def train_model_pytorch(model, X_train, y_train, num_epochs=100, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train.view(-1, 1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def train_model_sklearn(model, X_train, y_train):
    model.fit(X_train, y_train)

def evaluate_model_pytorch(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test)
        y_pred = y_pred_tensor.cpu().numpy().flatten()
        y_test = y_test.cpu().numpy().flatten()

        if y_pred.shape != y_test.shape:
            raise ValueError(f"Shape mismatch: y_pred has shape {y_pred.shape}, but y_test has shape {y_test.shape}")

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return mse, rmse, mae, r2

def evaluate_model_sklearn(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

def future_aqi_ml(model, city, days=3650):
    future_aqi = []
    for _ in range(days):
        input_data = np.random.rand(1, 3)
        if isinstance(model, nn.Module):
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)  # seq_len is set to 1
            output = model(input_tensor)
            future_aqi.append(output.item())
        else:
            output = model.predict(input_data)
            future_aqi.append(output[0])
    return pd.DataFrame({'Date': pd.date_range(start='2023-01-01', periods=days, freq='D'), 'City': [city] * days, 'AQI': future_aqi})

cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow']
for name, model in models.items():
    if name in ['LSTM', 'GRU']:
        train_model_pytorch(model, X_train_tensor_lstm, y_train_tensor)
        mse, rmse, mae, r2 = evaluate_model_pytorch(model, X_test_tensor_lstm, y_test_tensor)
    elif name == 'SARIMA':
        results = model.fit()
        predictions = results.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        predictions.index = y_test.index  # Align indices
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
    else:
        train_model_sklearn(model, X_train_scaled, y_train)
        mse, rmse, mae, r2 = evaluate_model_sklearn(model, X_test_scaled, y_test)
    
    print(f'\nEvaluation Metrics for {name}:')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R-squared: {r2:.2f}')

# Feature importance score (bar graph) for tree-based models
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        plt.bar(X.columns, model.feature_importances_)
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title(f'Feature Importance for {name}')
        plt.show()

# Plot of true vs predicted values for each model
for name, model in models.items():
    if name in ['LSTM', 'GRU']:
        y_pred_tensor = model(X_test_tensor_lstm)
        y_pred = y_pred_tensor.detach().cpu().numpy().flatten()
    elif name == 'SARIMA':
        y_pred = results.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        y_pred.index = y_test.index  # Align indices
    else:
        y_pred = model.predict(X_test_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'True vs Predicted Values for {name}')
    plt.grid(True)
    plt.show()


# Plot AQI parameter comparison for each city on a monthly basis
for city in cities:
    for name, model in models.items():
        if name == 'SARIMA':
            future_aqi_df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=3650, freq='D'),
                'City': [city] * 3650,
                'AQI': results.predict(start=0, end=3650 - 1)
            })
        else:
            future_aqi_df = future_aqi_ml(model, city, days=3650)
        future_aqi_df['Month'] = future_aqi_df['Date'].dt.month
        future_aqi_df['Year'] = future_aqi_df['Date'].dt.year
        future_aqi_monthly_avg = future_aqi_df.groupby(['Year', 'Month'])['AQI'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=future_aqi_monthly_avg, x='Month', y='AQI', hue='Year')
        plt.xlabel('Month')
        plt.ylabel('AQI')
        plt.title(f'Monthly AQI Predictions for {city} till 2030 using {name}')
        plt.grid(True)
        plt.legend()
        plt.show()

# Plot AQI parameter comparison for all cities on a monthly basis
for name, model in models.items():
    future_aqi_dfs = []
    for city in cities:
        if name == 'SARIMA':
            future_aqi_df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=3650, freq='D'),
                'City': [city] * 3650,
                'AQI': results.predict(start=0, end=3650 - 1)
            })
        else:
            future_aqi_df = future_aqi_ml(model, city, days=3650)
        future_aqi_df['Month'] = future_aqi_df['Date'].dt.month
        future_aqi_df['Year'] = future_aqi_df['Date'].dt.year
        future_aqi_df['City'] = city
        future_aqi_dfs.append(future_aqi_df)
    future_aqi_all_cities = pd.concat(future_aqi_dfs)
    future_aqi_monthly_avg = future_aqi_all_cities.groupby(['Year', 'Month', 'City'])['AQI'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=future_aqi_monthly_avg, x='Month', y='AQI', hue='City')
    plt.xlabel('Month')
    plt.ylabel('AQI')
    plt.title(f'Monthly AQI Predictions for all cities till 2030 using {name}')
    plt.grid(True)
    plt.legend()
    plt.show()

# Plot AQI parameter comparison for all cities on a yearly basis
for name, model in models.items():
    future_aqi_dfs = []
    for city in cities:
        if name == 'SARIMA':
            future_aqi_df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=3650, freq='D'),
                'City': [city] * 3650,
                'AQI': results.predict(start=0, end=3650 - 1)
            })
        else:
            future_aqi_df = future_aqi_ml(model, city, days=3650)
        future_aqi_df['Year'] = future_aqi_df['Date'].dt.year
        future_aqi_df['City'] = city
        future_aqi_dfs.append(future_aqi_df)
    future_aqi_all_cities = pd.concat(future_aqi_dfs)
    future_aqi_yearly_avg = future_aqi_all_cities.groupby(['Year', 'City'])['AQI'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=future_aqi_yearly_avg, x='Year', y='AQI', hue='City')
    plt.xlabel('Year')
    plt.ylabel('AQI')
    plt.title(f'Yearly AQI Predictions for all cities till 2030 using {name}')
    plt.grid(True)
    plt.legend()
    plt.show()

# Comparison graph of each pollutant feature and AQI from start to future prediction for 10 years
for city in cities:
    for name, model in models.items():
        if name == 'SARIMA':
            future_aqi_df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=3650, freq='D'),
                'City': [city] * 3650,
                'AQI': results.predict(start=0, end=3650 - 1)
            })
        else:
            future_aqi_df = future_aqi_ml(model, city, days=3650)
        future_aqi_df['Year'] = future_aqi_df['Date'].dt.year
        future_aqi_df['City'] = city
        future_aqi_dfs.append(future_aqi_df)
    future_aqi_all_cities = pd.concat(future_aqi_dfs)
    future_aqi_yearly_avg = future_aqi_all_cities.groupby(['Year', 'City'])['AQI'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=future_aqi_yearly_avg, x='Year', y='AQI', hue='City')
    plt.xlabel('Year')
    plt.ylabel('AQI')
    plt.title(f'Yearly AQI Predictions for all cities till 2030 using {name}')
    plt.grid(True)
    plt.legend()
    plt.show()

# Accuracy of each model
for name, model in models.items():
    if name in ['LSTM', 'GRU']:
        mse, rmse, mae, r2 = evaluate_model_pytorch(model, X_test_tensor_lstm, y_test_tensor)
    elif name == 'SARIMA':
        predictions = results.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
        predictions.index = y_test.index  # Align indices
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
    else:
        mse, rmse, mae, r2 = evaluate_model_sklearn(model, X_test_scaled, y_test)
    
    print(f'\nAccuracy Metrics for {name}:')
    print(f'MSE: {mse:.2f}')
    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'R-squared: {r2:.2f}')

# Comparison of cities with each other
for name, model in models.items():
    future_aqi_dfs = []
    for city in cities:
        if name == 'SARIMA':
            future_aqi_df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=3650, freq='D'),
                'City': [city] * 3650,
                'AQI': results.predict(start=0, end=3650 - 1)
            })
        else:
            future_aqi_df = future_aqi_ml(model, city, days=3650)
        future_aqi_df['Year'] = future_aqi_df['Date'].dt.year
        future_aqi_df['City'] = city
        future_aqi_dfs.append(future_aqi_df)
    future_aqi_all_cities = pd.concat(future_aqi_dfs)
    future_aqi_yearly_avg = future_aqi_all_cities.groupby(['Year', 'City'])['AQI'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=future_aqi_yearly_avg, x='Year', y='AQI', hue='City')
    plt.xlabel('Year')
    plt.ylabel('AQI')
    plt.title(f'Yearly AQI Predictions for all cities till 2030 using {name}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Filter data for the years 2019-2022
data_filtered_covid = data[(data.index.year >= 2019) & (data.index.year <= 2022)]

# Calculate monthly average of AQI
aqi_monthly_avg_covid = data_filtered_covid.resample('M').mean()

# Plot monthly average of AQI for the years 2019-2022
plt.figure(figsize=(12, 6))
sns.lineplot(data=aqi_monthly_avg_covid, x=aqi_monthly_avg_covid.index, y='AQI')
plt.xlabel('Date')
plt.ylabel('Average Monthly AQI')
plt.title('Monthly Average AQI (2019-2022)')
plt.grid(True)
plt.show()
