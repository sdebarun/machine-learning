import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your CSV file
data = pd.read_csv('cvd_2d_crystal_data.csv')

# Select features and target
features = ['TMo', 'TS', 'Flow', 's']
target = 'Average size'

X = data[features].values
y = data[target].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Metrics
print("Coefficients:", regressor.coef_)
print("Intercept:", regressor.intercept_)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R²:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Average Size")
plt.ylabel("Predicted Average Size")
plt.title("Actual vs Predicted Average Size")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# Bar plots for each feature
bins = 5
plt.figure(figsize=(16, 10))
for i, feature in enumerate(features, start=1):
    data['bin'] = pd.cut(data[feature], bins=bins)
    grouped = data.groupby('bin')[target].mean()
    plt.subplot(2, 2, i)
    grouped.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Average Size vs Binned {feature}')
    plt.xlabel(f'{feature} bins')
    plt.ylabel('Average Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
data.drop(columns=['bin'], inplace=True)
plt.show()



# For growth parameters (like Fig S9)
growth_params = ["TMo", "TS", "Flow", "s"]
growth_corr = data[growth_params].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(
    growth_corr, annot=True, cmap="Blues", square=True, vmin=-1, vmax=1,
    linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation Matrix of CVD Growth Parameters")
plt.show()

# For morphology parameters (like Fig S10)
morphology_params = [
     "Coverage",
    "Average size",
    "Edge length",
    "Nucleation rate",
]
morph_corr = data[morphology_params].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    morph_corr, annot=True, cmap="Greens", square=True, vmin=-1, vmax=1,
    linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation Matrix of Morphology Parameters")
plt.show()
