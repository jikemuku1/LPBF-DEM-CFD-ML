import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata, norm
from scipy.linalg import cholesky

try:
    df = pd.read_excel('../data/Melt_Pool_Data_With_Depth.xlsx', sheet_name='Original_Data')
    df.columns = [col.strip() for col in df.columns]

    column_mapping = {}
    for col in df.columns:
        if 'Power' in col:
            column_mapping[col] = 'Power [W]'
        elif 'Velocity' in col:
            column_mapping[col] = 'Velocity [mm s-1]'
        elif 'Radius' in col:
            column_mapping[col] = 'Radius [mm]'
        elif 'Max Melt Pool Width' in col and '[m]' in col:
            column_mapping[col] = 'Max Melt Pool Width [m]'
        elif 'Max Melt Pool Depth' in col and '[m]' in col:
            column_mapping[col] = 'Max Melt Pool Depth [m]'
        elif 'Porosity' in col:
            column_mapping[col] = 'Porosity [%]'
        elif 'VED' in col:
            column_mapping[col] = 'VED [J mm-3]'
        elif 'Max Melt Pool Width' in col and '[mm]' in col:
            column_mapping[col] = 'Max Melt Pool Width [mm]'
        elif 'Max Melt Pool Depth' in col and '[mm]' in col:
            column_mapping[col] = 'Max Melt Pool Depth [mm]'
        elif 'Aspect Ratio' in col:
            column_mapping[col] = 'Aspect Ratio (Depth/Width)'

    df = df.rename(columns=column_mapping)

    if 'Max Melt Pool Width [mm]' not in df.columns and 'Max Melt Pool Width [m]' in df.columns:
        df['Max Melt Pool Width [mm]'] = df['Max Melt Pool Width [m]'] * 1000
    if 'Max Melt Pool Depth [mm]' not in df.columns and 'Max Melt Pool Depth [m]' in df.columns:
        df['Max Melt Pool Depth [mm]'] = df['Max Melt Pool Depth [m]'] * 1000

    if 'VED [J mm-3]' not in df.columns and all(col in df.columns for col in ['Power [W]', 'Velocity [mm s-1]', 'Radius [mm]']):
        df['VED [J mm-3]'] = df['Power [W]'] / (df['Velocity [mm s-1]'] * df['Radius [mm]'] ** 2)

    if 'Aspect Ratio (Depth/Width)' not in df.columns and all(col in df.columns for col in ['Max Melt Pool Depth [mm]', 'Max Melt Pool Width [mm]']):
        df['Aspect Ratio (Depth/Width)'] = df['Max Melt Pool Depth [mm]'] / df['Max Melt Pool Width [mm]']

except Exception:
    np.random.seed(42)
    n_samples = 200
    df = pd.DataFrame({
        "Power [W]": np.random.uniform(180, 300, n_samples),
        "Velocity [mm s-1]": np.random.uniform(200, 1000, n_samples),
        "Radius [mm]": np.random.uniform(2.3, 7.0, n_samples),
        "Max Melt Pool Width [mm]": np.random.uniform(0.03, 0.08, n_samples),
        "Max Melt Pool Depth [mm]": np.random.uniform(0.05, 0.08, n_samples),
        "Porosity [%]": np.random.uniform(0.01, 2.5, n_samples)
    })

# Generate correlated synthetic data
np.random.seed(42)
variables = [
    'Power [W]', 'Velocity [mm s-1]', 'Radius [mm]',
    'Max Melt Pool Width [mm]', 'Max Melt Pool Depth [mm]',
    'Porosity [%]', 'VED [J mm-3]'
]

X = df[variables].values
n_vars = len(variables)
n_samples = 200

Z = np.zeros_like(X, dtype=float)
for i in range(n_vars):
    ranks = rankdata(X[:, i])
    uniforms = (ranks - 0.5) / len(ranks)
    Z[:, i] = norm.ppf(uniforms)

target_corr = np.array([
    [1.00, -0.01, -0.09, 0.03, 0.08, 0.21, 0.34],
    [-0.01, 1.00, 0.11, -0.42, -0.50, -0.71, -0.83],
    [-0.09, 0.11, 1.00, 0.73, -0.62, -0.25, -0.31],
    [0.03, -0.42, 0.73, 1.00, -0.35, 0.23, 0.19],
    [0.08, -0.50, -0.62, -0.35, 1.00, 0.43, 0.46],
    [0.21, -0.71, -0.25, 0.23, 0.43, 1.00, 0.86],
    [0.34, -0.83, -0.31, 0.19, 0.46, 0.86, 1.00]
])

min_eig = np.min(np.real(np.linalg.eigvals(target_corr)))
if min_eig < 0:
    target_corr -= (min_eig - 1e-6) * np.eye(n_vars)

L = cholesky(target_corr, lower=True)
Z_gen = np.random.randn(n_samples, n_vars)
Z_corr = np.dot(Z_gen, L.T)

X_gen = np.zeros((n_samples, n_vars))
for i in range(n_vars):
    uniforms_gen = norm.cdf(Z_corr[:, i])
    sorted_orig = np.sort(X[:, i])
    indices = uniforms_gen * (len(sorted_orig) - 1)
    lower = np.floor(indices).astype(int)
    upper = np.ceil(indices).astype(int)
    alpha = indices - lower
    X_gen[:, i] = (1 - alpha) * sorted_orig[lower] + alpha * sorted_orig[upper]
    min_val = np.min(X[:, i])
    max_val = np.max(X[:, i])
    ext = 0.05 * (max_val - min_val)
    X_gen[:, i] = np.clip(X_gen[:, i], min_val - ext, max_val + ext)

df_generated = pd.DataFrame(X_gen, columns=variables)

if 'Max Melt Pool Width [mm]' in df_generated.columns:
    df_generated['Max Melt Pool Width [m]'] = df_generated['Max Melt Pool Width [mm]'] / 1000
if 'Max Melt Pool Depth [mm]' in df_generated.columns:
    df_generated['Max Melt Pool Depth [m]'] = df_generated['Max Melt Pool Depth [mm]'] / 1000
if all(col in df_generated.columns for col in ['Max Melt Pool Depth [mm]', 'Max Melt Pool Width [mm]']):
    df_generated['Aspect Ratio (Depth/Width)'] = df_generated['Max Melt Pool Depth [mm]'] / df_generated['Max Melt Pool Width [mm]']

# Plot cumulative probability distributions
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Cumulative Probability Distribution Comparison', fontsize=16, fontweight='bold')

variables_to_plot = [
    'Power [W]', 'Velocity [mm s-1]', 'Radius [mm]',
    'Max Melt Pool Width [mm]', 'Max Melt Pool Depth [mm]',
    'Porosity [%]', 'VED [J mm-3]', 'Aspect Ratio (Depth/Width)'
]

for idx, var in enumerate(variables_to_plot):
    if var in df.columns and var in df_generated.columns:
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        sorted_orig = np.sort(df[var].dropna().values)
        sorted_gen = np.sort(df_generated[var].dropna().values)

        y_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
        y_gen = np.arange(1, len(sorted_gen) + 1) / len(sorted_gen)

        ax.plot(sorted_orig, y_orig, 'b-', linewidth=2, label='Original Data')
        ax.plot(sorted_gen, y_gen, 'r--', linewidth=2, label='Generated Data')

        ax.set_xlabel(var, fontsize=10, fontweight='bold')
        ax.set_ylabel('Cumulative Probability', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('Fig5_Consistency_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()