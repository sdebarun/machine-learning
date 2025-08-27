import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (adjust filename if needed)
dataset = pd.read_csv('cvd_2d_crystal_data.csv')

# Select independent variables and target variable explicitly
features = ['TMo', 'TS', 'Flow', 's']
target = 'Average size'

X = dataset[features].values
y = dataset[target].values

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the multiple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on test set
y_pred = regressor.predict(X_test)

# Print model coefficients and intercept
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Plot Actual vs Predicted values
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Average Size")
plt.ylabel("Predicted Average Size")
plt.title("Actual vs Predicted Average Size")
plt.tight_layout()
plt.show()

# -----------------------------------------------
# Bar Graphs: Average Size vs Binned Independent Variables
# -----------------------------------------------

bins = 5  # Number of bins to group each feature

plt.figure(figsize=(16, 10))

for i, feature in enumerate(features, start=1):
    # Bin feature values into intervals
    dataset['bin'] = pd.cut(dataset[feature], bins=bins)
    
    # Compute average of target 'Average size' for each bin
    grouped = dataset.groupby('bin')[target].mean()
    
    # Plot bar graph
    plt.subplot(2, 2, i)
    grouped.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Average {target} vs Binned {feature}')
    plt.xlabel(f'{feature} bins')
    plt.ylabel(f'Average {target}')
    plt.xticks(rotation=45)
    plt.tight_layout()

# Remove the temporary 'bin' column to clean up dataset
dataset.drop(columns=['bin'], inplace=True)

plt.show()
