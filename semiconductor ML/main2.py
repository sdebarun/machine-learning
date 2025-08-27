# ===============================
# Multiple Linear Regression with Recursive Feature Elimination (RFE) and Bar Plots
# ===============================

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 2. Load Dataset
data = pd.read_csv("cvd_2d_crystal_data.csv")  # <-- Change this CSV file name if needed
print("Data preview:")
print(data.head())

# 3. Define Features and Target
target_column = "Average size"  # Change if needed
X = data.drop(columns=[target_column,"#"])
y = data[target_column]

# 4. Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. Initialize linear regression model
model = LinearRegression()

# 7. Recursive Feature Elimination (select top n features)
n_features_to_select = 3  # Customize this as needed
rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
rfe.fit(X_train, y_train)

# 8. Features selected by RFE
selected_features = X.columns[rfe.support_].tolist()
print(f"Selected features by RFE: {selected_features}")

# 9. Train model using selected features only
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)
model.fit(X_train_rfe, y_train)

# 10. Make predictions and evaluate
y_pred = model.predict(X_test_rfe)

print("\nModel performance with RFE-selected features:")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R² score: {r2_score(y_test, y_pred):.4f}")

# 11. Plot actual vs predicted values
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (RFE-selected features)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

# -----------------------------------------------
# Bar Graphs: Target vs Binned Independent Variables
# -----------------------------------------------

bins = 5  # Number of bins for grouping each feature

plt.figure(figsize=(16, 10))

for i, feature in enumerate(X.columns, start=1):
    # Bin feature values into intervals
    data['bin'] = pd.cut(data[feature], bins=bins)
    
    # Compute average of target variable for each bin
    grouped = data.groupby('bin')[target_column].mean()
    
    # Plot bar graph
    plt.subplot(3, 3, i)
    grouped.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Average {target_column} vs {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel(f'Average {target_column}')
    plt.xticks(rotation=45)
    plt.tight_layout()

# Remove temporary 'bin' column after plotting
data.drop(columns=['bin'], inplace=True)

plt.show()
