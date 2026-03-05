import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

t0 = time.time()

# [1] Load dataset
print("\n[1/7] Loading dataset...")
df = pd.read_excel('../data/Melt_Pool_Data_With_Depth.xlsx')

# [2] Feature engineering
def create_comprehensive_features(df):
    power = df["Power [W]"].values
    velocity = df["Velocity [mm s-1]"].values
    radius = df["Radius [mm]"].values


    # ---------------------------------------------------------------
    df['Linear_Energy_Density']      = power / velocity
    df['Area_Energy_Density']        = power / (velocity * radius)
    df['Volumetric_Energy_Density']  = power / (velocity * radius * 0.1)
    df['Specific_Energy']            = power / (velocity * radius**2)
    df['Energy_Flux']                = power / (np.pi * radius**2)
    df['Beam_Area']                  = np.pi * radius**2
    df['Beam_Circumference']         = 2 * np.pi * radius
    df['Beam_Diameter']              = 2 * radius
    df['Aspect_Ratio_Energy']        = power / (velocity * radius)
    df['Power_per_Area']             = power / (np.pi * radius**2)
    df['Power_per_Radius']           = power / radius
    df['Intensity']                  = power / (np.pi * radius**2)
    df['Specific_Power']             = power / radius
    df['Stability_Index']            = (power * radius) / velocity
    df['Process_Efficiency']         = power / (velocity * radius**3)
    df['Thermal_Gradient']           = power / radius**2
    df['Melting_Parameter']          = power * radius / velocity
    df['Heat_Affected_Zone']         = radius**2 / velocity
    df['Radius_Velocity_Ratio']      = radius / velocity
    df['Power_Radius_Ratio']         = power / radius
    df['Energy_Concentration']       = power / radius**2
    df['Dimensionless_Energy']       = (power * radius) / velocity**2
    df['Power_Radius_Interaction']   = power * radius
    df['Velocity_Radius_Interaction']= velocity * radius
    df['Power_Velocity_Radius']      = power * velocity * radius
    df['P_V_R_Combined']             = power / (velocity * radius)
    df['Log_Power']                  = np.log(power + 1e-6)
    df['Log_Velocity']               = np.log(velocity + 1e-6)
    df['Log_Radius']                 = np.log(radius + 1e-6)
    df['Radius_Squared']             = radius**2
    df['Inverse_Radius']             = 1.0 / (radius + 1e-6)
    df['Energy_Density_Radius']      = power / (velocity * radius**3)
    df['Conduction_Parameter']       = power / (velocity * radius)
    df['Penetration_Ratio']          = power / (velocity * radius**0.5)
    df['Fusion_Parameter']           = (power * radius) / velocity

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

df_eng = create_comprehensive_features(df)

TARGET_COLS = ['Max Melt Pool Width [mm]', 'Max Melt Pool Depth [mm]', 'Porosity [%]']
feature_columns = [c for c in df_eng.columns if c not in TARGET_COLS]

X_raw = df_eng[feature_columns]
y_width = df_eng['Max Melt Pool Width [mm]']
y_depth = df_eng['Max Melt Pool Depth [mm]']
y_porosity = df_eng['Porosity [%]']

# [3] Outlier removal (IQR)
def remove_outliers_iqr(y):
    Q1, Q3 = np.percentile(y, 25), np.percentile(y, 75)
    IQR = Q3 - Q1
    mask = (y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)
    return mask

final_mask = (
    remove_outliers_iqr(y_width) &
    remove_outliers_iqr(y_depth) &
    remove_outliers_iqr(y_porosity)
)

X = X_raw[final_mask].reset_index(drop=True)
y_width = y_width[final_mask].reset_index(drop=True)
y_depth = y_depth[final_mask].reset_index(drop=True)
y_porosity = y_porosity[final_mask].reset_index(drop=True)

# [4] Normalization
scaler_X = StandardScaler()
scaler_y_width = MinMaxScaler()
scaler_y_depth = MinMaxScaler()
scaler_y_porosity = MinMaxScaler()

X_n = scaler_X.fit_transform(X)
yw_n = scaler_y_width.fit_transform(y_width.values.reshape(-1, 1)).ravel()
yd_n = scaler_y_depth.fit_transform(y_depth.values.reshape(-1, 1)).ravel()
yp_n = scaler_y_porosity.fit_transform(y_porosity.values.reshape(-1, 1)).ravel()

# [5] GPR with ARD kernels
n_features = X_n.shape[1]

kernels = [
    C(1.0, (1e-3, 1e2)) * RBF(length_scale=np.ones(n_features),
                              length_scale_bounds=(1e-3, 1e2)),
    C(1.0, (1e-3, 1e2)) * Matern(nu=2.5,
                                 length_scale=np.ones(n_features),
                                 length_scale_bounds=(1e-3, 1e2))
]

param_grid = {
    "kernel": kernels,
    "alpha": [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5],
    "n_restarts_optimizer": [15],
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

def train_gpr(X, y, name):
    print(f"  Training GPR for {name} with ARD...")
    t_start = time.time()
    gpr = GaussianProcessRegressor(normalize_y=True)
    gs = GridSearchCV(
        gpr,
        param_grid,
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        refit=True
    )
    gs.fit(X, y)
    best = gs.best_estimator_
    y_pred = cross_val_predict(best, X, y, cv=kf, n_jobs=-1)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    t_elapsed = time.time() - t_start
    return best, y_pred, mse, r2, mape, t_elapsed

print("\n[6/7] Training three GPR models...")
m_w, yp_w, mse_w, r2_w, mape_w, t_w = train_gpr(X_n, yw_n, "Width")
m_d, yp_d, mse_d, r2_d, mape_d, t_d = train_gpr(X_n, yd_n, "Depth")
m_p, yp_p, mse_p, r2_p, mape_p, t_p = train_gpr(X_n, yp_n, "Porosity")

# [6] Results
print("\n[7/7] Results (10-fold CV):")
print(f"{'':12s}  {'R²':>8s}  {'MSE':>12s}  {'MAPE':>8s}")
print("─" * 46)
print(f"{'Width':12s}  {r2_w:8.4f}  {mse_w:12.4e}  {mape_w:8.4f}")
print(f"{'Depth':12s}  {r2_d:8.4f}  {mse_d:12.4e}  {mape_d:8.4f}")
print(f"{'Porosity':12s}  {r2_p:8.4f}  {mse_p:12.4e}  {mape_p:8.4f}")
print(f"\n✓ Individual Training Times:")
print(f"  Width training time: {t_w:.2f} s")
print(f"  Depth training time: {t_d:.2f} s")
print(f"  Porosity training time: {t_p:.2f} s")
print(f"\n  Average R² = {np.mean([r2_w, r2_d, r2_p]):.4f}")

# [7] Plot (Fig 13a-style)
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    yw_n, yp_w, marker='o', s=200, color='lightblue',
    edgecolor='black', linewidths=1.5, alpha=0.9,
    label=f'Molten Pool Width  (R²: {r2_w:.4f})', zorder=3
)
ax.scatter(
    yd_n, yp_d, marker='s', s=200, color='lightgreen',
    edgecolor='black', linewidths=1.5, alpha=0.9,
    label=f'Molten Pool Depth  (R²: {r2_d:.4f})', zorder=3
)
ax.scatter(
    yp_n, yp_p, marker='^', s=200, color='orange',
    edgecolor='black', linewidths=1.5, alpha=0.9,
    label=f'Porosity            (R²: {r2_p:.4f})', zorder=3
)

ax.plot([0, 1], [0, 1], 'r--', linewidth=3.5, label='Ideal y = x', zorder=2)
ax.set_xlabel('True Values (Normalized)', fontsize=20, fontweight='bold')
ax.set_ylabel('Predicted Values (Normalized)', fontsize=20, fontweight='bold')
ax.tick_params(labelsize=18)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(prop={'size': 15, 'weight': 'bold'}, loc='upper left',
          framealpha=1, edgecolor='black')
ax.grid(False)
plt.tight_layout()
plt.savefig('Fig13a_GPR_v3_final.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Figure saved → Fig13a_GPR_v3_final.png")

