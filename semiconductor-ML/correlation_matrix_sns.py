import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data loading
data = pd.read_csv('cvd_2d_crystal_data.csv')

# For growth parameters
growth_params = ["TMo", "TS", "Flow", "s"]
growth_corr = data[growth_params].corr()

plt.figure(figsize=(6, 5))
sns.heatmap(
    growth_corr, annot=True, cmap="Blues", square=True, vmin=-1, vmax=1,
    linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation Matrix of CVD Growth Parameters")
plt.show()

# For morphology parameters
morphology_params = [
     "Coverage",
    "Average size",
    "Edge length",
    "Nucleation density",
    "Nucleation rate",
]
morph_corr = data[morphology_params].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    morph_corr, annot=True, cmap="Greens", square=True, vmin=-1, vmax=1,
    linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation Matrix of Morphology Parameters")
plt.show()