import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('cvd_2d_crystal_data.csv')

# ------ Growth Parameters Correlation Matrix -----
growth_params = ['TMo', 'TS', 'Flow', 's']
growth_corr = data[growth_params].corr()

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(growth_corr, cmap='Blues', vmin=-1, vmax=1)

# Show all ticks and label them
ax.set_xticks(np.arange(len(growth_params)))
ax.set_yticks(np.arange(len(growth_params)))
ax.set_xticklabels(growth_params)
ax.set_yticklabels(growth_params)

# Rotate the tick labels and set alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Annotate the heatmap with the correlation coefficients
for i in range(len(growth_params)):
    for j in range(len(growth_params)):
        ax.text(j, i, f"{growth_corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")

# Add a colorbar
fig.colorbar(im, ax=ax, shrink=0.8)
plt.title("Correlation Matrix of CVD Growth Parameters")
plt.tight_layout()
plt.show()


# ------ Morphology Parameters Correlation Matrix -----
morphology_params = ['Coverage', 'Average size', 'Edge length', 'Nucleation density', 'Nucleation rate']
morph_corr = data[morphology_params].corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(morph_corr, cmap='Greens', vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(morphology_params)))
ax.set_yticks(np.arange(len(morphology_params)))
ax.set_xticklabels(morphology_params)
ax.set_yticklabels(morphology_params)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(len(morphology_params)):
    for j in range(len(morphology_params)):
        ax.text(j, i, f"{morph_corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")

fig.colorbar(im, ax=ax, shrink=0.8)
plt.title("Correlation Matrix of Morphology Parameters")
plt.tight_layout()
plt.show()