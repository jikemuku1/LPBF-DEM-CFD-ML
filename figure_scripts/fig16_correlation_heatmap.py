import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'

# Load data
df = pd.read_excel("../data/Melt_Pool_Data_With_Depth.xlsx", sheet_name="Original_Data")

# Unit conversion
df["Molten Pool Width [mm]"] = df["Max Melt Pool Width [mm]"]
df["Molten Pool Depth [mm]"] = df["Max Melt Pool Depth [mm]"]

# Recalculate VED
denominator = (df["Velocity [mm s-1]"] * df["Molten Pool Width [mm]"] * df["Molten Pool Depth [mm]"])
denominator = denominator.replace(0, np.nan)
df["VED [J mm-3]"] = 0.4 * df["Power [W]"] / denominator
df = df.dropna(subset=["VED [J mm-3]"])

# Select and sort variables
selected_vars = [
    'Power [W]',
    'Velocity [mm s-1]',
    'Radius [mm]',
    'Molten Pool Width [mm]',
    'Molten Pool Depth [mm]',
    'Porosity [%]',
    'VED [J mm-3]',
]

display_labels = [
    'Power [W]',
    'Velocity [mm s⁻¹]',
    'Radius [mm]',
    'Width [mm]',
    'Depth [mm]',
    'Porosity [%]',
    'VED [J mm⁻³]',
]

corr = df[selected_vars].corr()
corr.index = display_labels
corr.columns = display_labels

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 8))

hm = sns.heatmap(
    corr,
    ax=ax,
    annot=True,
    fmt=".2f",
    cmap='PuOr',
    vmin=-1, vmax=1,
    annot_kws={"size": 12, "weight": "bold", "family": "Times New Roman"},
    linewidths=0.5,
    linecolor='white',
    cbar=True,
    cbar_kws={"shrink": 0.85, "pad": 0.02},
    square=True,
)

# Add dividing lines
ax.axhline(y=3, color='#2E75B6', linewidth=2.5, linestyle='-')
ax.axvline(x=3, color='#2E75B6', linewidth=2.5, linestyle='-')
ax.axhline(y=6, color='#2E75B6', linewidth=1.5, linestyle='--')
ax.axvline(x=6, color='#2E75B6', linewidth=1.5, linestyle='--')

# Colorbar formatting
cbar = hm.collections[0].colorbar
cbar.set_label('Pearson Correlation Coefficient', fontsize=12, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=11)
for tick in cbar.ax.get_yticklabels():
    tick.set_fontfamily('Times New Roman')
    tick.set_fontweight('bold')

# Axis tick labels
ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=12, fontweight='bold')
ax.set_yticklabels(display_labels, rotation=0, ha='right', fontsize=12, fontweight='bold')

# Title
ax.set_title('Reorganized Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig("Fig16_Correlation_Heatmap.png", dpi=300, bbox_inches="tight")
plt.close()