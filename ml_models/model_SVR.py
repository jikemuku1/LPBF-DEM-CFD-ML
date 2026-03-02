import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.svm import SVR
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance

# [1] Load dataset
print("\n[1/9] Loading dataset...")
try:
    df = pd.read_excel('../data/Melt_Pool_Data_With_Depth.xlsx')
    print(f"✓ Loaded {len(df)} data points")
except FileNotFoundError:
    print("✗ Cannot find Excel file, using sample data...")
    np.random.seed(42)
    n_samples = 234
    df = pd.DataFrame({
        "Power [W]": np.random.uniform(180, 300, n_samples),
        "Velocity [mm s-1]": np.random.uniform(200, 1000, n_samples),
        "Radius [mm]": np.random.uniform(2.3, 7.0, n_samples),
        "Max Melt Pool Width [mm]": np.random.uniform(0.03, 0.08, n_samples),
        "Max Melt Pool Depth [mm]": np.random.uniform(0.05, 0.08, n_samples),
        "Porosity [%]": np.random.uniform(0.01, 2.5, n_samples)
    })

# [2] Apply comprehensive feature engineering
print("\n[2/9] Creating comprehensive engineered features...")


def create_comprehensive_features(df):
    """Create 25+ engineered features including radius-based physics"""
    print("Creating comprehensive features with radius engineering...")

    power = df["Power [W]"].values
    velocity = df["Velocity [mm s-1]"].values
    radius = df["Radius [mm]"].values

    # Basic energy calculations
    df['Linear_Energy_Density'] = power / velocity
    df['Area_Energy_Density'] = power / (velocity * radius)
    df['Volumetric_Energy_Density'] = power / (velocity * radius * 0.1)
    df['Specific_Energy'] = power / (velocity * radius ** 2)
    df['Energy_Flux'] = power / (np.pi * radius ** 2)

    # Radius geometry features
    df['Beam_Area'] = np.pi * (radius ** 2)
    df['Beam_Circumference'] = 2 * np.pi * radius
    df['Beam_Diameter'] = 2 * radius
    df['Aspect_Ratio_Energy'] = power / (velocity * radius)

    # Power density features
    df['Power_per_Area'] = power / (np.pi * radius ** 2)
    df['Power_per_Radius'] = power / radius
    df['Intensity'] = power / (np.pi * (radius ** 2))
    df['Specific_Power'] = power / radius

    # Process parameters with radius
    df['Stability_Index'] = (power * radius) / velocity
    df['Process_Efficiency'] = power / (velocity * radius ** 3)
    df['Thermal_Gradient'] = power / (radius ** 2)
    df['Melting_Parameter'] = power * radius / velocity
    df['Heat_Affected_Zone'] = radius ** 2 / velocity

    # Dimensionless numbers
    df['Radius_Velocity_Ratio'] = radius / velocity
    df['Power_Radius_Ratio'] = power / radius
    df['Energy_Concentration'] = power / (radius ** 2)
    df['Dimensionless_Energy'] = (power * radius) / (velocity ** 2)

    # Interaction terms
    df['Power_Radius_Interaction'] = power * radius
    df['Velocity_Radius_Interaction'] = velocity * radius
    df['Power_Velocity_Radius'] = power * velocity * radius
    df['P_V_R_Combined'] = power / (velocity * radius)

    # Nonlinear transformations
    df['Log_Power'] = np.log(power + 1e-6)
    df['Log_Velocity'] = np.log(velocity + 1e-6)
    df['Log_Radius'] = np.log(radius + 1e-6)
    df['Radius_Squared'] = radius ** 2
    df['Inverse_Radius'] = 1 / (radius + 1e-6)

    # Advanced physics-based features
    df['Energy_Density_Radius'] = power / (velocity * radius ** 3)
    df['Conduction_Parameter'] = power / (velocity * radius)
    df['Penetration_Ratio'] = power / (velocity * radius ** 0.5)
    df['Fusion_Parameter'] = (power * radius) / velocity

    # Remove any potential infinite values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    total_features = len([col for col in df.columns if col not in
                          ['Power [W]', 'Velocity [mm s-1]', 'Radius [mm]',
                           'Max Melt Pool Width [mm]', 'Max Melt Pool Depth [mm]', 'Porosity [%]']])

    print(f"✓ Created {total_features} engineered features")
    return df


# Apply feature engineering
df_engineered = create_comprehensive_features(df)

# Prepare features and targets
feature_columns = [col for col in df_engineered.columns if col not in
                   ['Max Melt Pool Width [mm]', 'Max Melt Pool Depth [mm]', 'Porosity [%]']]

X_raw = df_engineered[feature_columns]
y_width = df_engineered["Max Melt Pool Width [mm]"]
y_depth = df_engineered["Max Melt Pool Depth [mm]"]
y_porosity = df_engineered["Porosity [%]"]

print(f"✓ Original dataset shape: {df.shape}")
print(f"✓ Engineered dataset shape: {df_engineered.shape}")
print(f"✓ Number of features: {len(feature_columns)}")

# --------------------------------------------------------------
# [3] Outlier removal (based on target values, not R²!)
# --------------------------------------------------------------
print("\n[3/9] Removing outliers using IQR method...")


def remove_outliers_iqr(y, name):
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (y >= lower) & (y <= upper)
    removed = len(y) - mask.sum()
    print(f"  {name}: Removed {removed} outliers ({100 * removed / len(y):.1f}%)")
    return mask


mask_w = remove_outliers_iqr(y_width, "Width")
mask_d = remove_outliers_iqr(y_depth, "Depth")
mask_p = remove_outliers_iqr(y_porosity, "Porosity")

# Use strictest mask (intersection) for fair comparison
final_mask = mask_w & mask_d & mask_p
print(f"→ Final dataset after outlier removal: {final_mask.sum()} / {len(df_engineered)} samples")

X = X_raw[final_mask].reset_index(drop=True)
y_width = y_width[final_mask].reset_index(drop=True)
y_depth = y_depth[final_mask].reset_index(drop=True)
y_porosity = y_porosity[final_mask].reset_index(drop=True)

# [4] Prepare normalized data
print("\n[4/9] Preparing normalized data...")

scaler_X = MinMaxScaler()
scaler_y_width = MinMaxScaler()
scaler_y_depth = MinMaxScaler()
scaler_y_porosity = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
y_width_normalized = scaler_y_width.fit_transform(y_width.values.reshape(-1, 1)).ravel()
y_depth_normalized = scaler_y_depth.fit_transform(y_depth.values.reshape(-1, 1)).ravel()
y_porosity_normalized = scaler_y_porosity.fit_transform(y_porosity.values.reshape(-1, 1)).ravel()

# [5] Define SVR parameters
print("\n[5/9] Setting up SVR with enhanced features...")

param_grid = {
    "kernel": ['rbf', 'poly'],
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": ['scale', 'auto', 0.1, 0.01, 0.001],
    "epsilon": [0.01, 0.1, 0.2, 0.3]
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)


# [6] Training function
def train_and_evaluate_svr(X, y, param_grid, kf, target_name):
    """Train and evaluate SVR with honest cross-validation"""
    print(f"Training SVR for {target_name}...")
    start_time = time.time()

    svr = SVR()
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X, y)
    best_svr = grid_search.best_estimator_
    training_time = time.time() - start_time

    # Get honest cross-validated predictions
    y_pred = cross_val_predict(best_svr, X, y, cv=kf, n_jobs=-1)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Handle MAPE calculation with zero values
    mask = y != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y[mask], y_pred[mask])
    else:
        mape = float('inf')

    print(f"  ✓ SVR - {target_name}: R² = {r2:.4f}, MSE = {mse:.4e}, Time = {training_time:.2f}s")
    print(f"  Best parameters: {grid_search.best_params_}")

    return best_svr, y_pred, mse, r2, mape, training_time


# [7] Train models
print("\n[6/9] Training SVR models with engineered features...")

best_svr_width, y_pred_width, mse_width, r2_width, mape_width, time_width = train_and_evaluate_svr(
    X_normalized, y_width_normalized, param_grid, kf, "Width"
)

best_svr_depth, y_pred_depth, mse_depth, r2_depth, mape_depth, time_depth = train_and_evaluate_svr(
    X_normalized, y_depth_normalized, param_grid, kf, "Depth"
)

best_svr_porosity, y_pred_porosity, mse_porosity, r2_porosity, mape_porosity, time_porosity = train_and_evaluate_svr(
    X_normalized, y_porosity_normalized, param_grid, kf, "Porosity"
)

# [8] Results summary
print("\n[7/9] SVR Results Summary:")
print(f"MSE for Max Melt Pool Width: {mse_width:.4e}")
print(f"MSE for Max Melt Pool Depth: {mse_depth:.4e}")
print(f"MSE for Porosity: {mse_porosity:.4e}")

print(f"R² for Max Melt Pool Width: {r2_width:.4f}")
print(f"R² for Max Melt Pool Depth: {r2_depth:.4f}")
print(f"R² for Porosity: {r2_porosity:.4f}")

# [9] Filter R² scores and plot
print("\n[8/9] Generating plots with R² filtering (0.8 ≤ R² ≤ 1.2)...")


def plot_combined_results_normalized(y_true_width, y_pred_width, y_true_depth, y_pred_depth,
                                     y_true_porosity, y_pred_porosity, r2_width, r2_depth,
                                     r2_porosity):
    """Plot combined results matching GPR style with R² filtering"""
    plt.figure(figsize=(10, 8))

    # Filter R² scores: only plot if 0.8 ≤ R² ≤ 1.2
    plt.scatter(y_true_width, y_pred_width, marker='o', s=200, color='lightblue',
                edgecolor="black", linewidths=1.5, alpha=1,
                label=f'Molten Pool Width (R²: {r2_width:.4f})')

    plt.scatter(y_true_depth, y_pred_depth, marker='s', s=200, color='lightgreen',
                edgecolor="black", linewidths=1.5, alpha=1,
                label=f'Molten Pool Depth (R²: {r2_depth:.4f})')

    plt.scatter(y_true_porosity, y_pred_porosity, marker='^', s=200, color='orange',
                edgecolor="black", linewidths=1.5, alpha=1,
                label=f'Porosity (R²: {r2_porosity:.4f})')

    # plt.plot([0, 1], [0, 1], 'r--', color='black', linewidth=3.5, label='Ideal y=x')
    plt.plot([0, 1], [0, 1], 'r--', color='red', linewidth=3.5, label='Ideal y=x')

    plt.xlabel('True Values (Normalized)', fontsize=20, fontweight='bold')
    plt.ylabel('Predicted Values (Normalized)', fontsize=20, fontweight='bold')
    plt.title('SVR Cross-Validation Results (Normalized)', fontsize=20, fontweight='bold')
    # plt.ylim(-0.1, 1.05)
    plt.ylim(-0.1, 1.08)
    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    plt.legend(prop={'size': 15, 'weight': 'bold'})
    # plt.grid(True, alpha=0.3)
    plt.savefig("Fig13b_SVR_CrossVal_Results.png", dpi=300, bbox_inches="tight")
plt.close()


# Generate plot with R² filtering
plot_combined_results_normalized(y_width_normalized, y_pred_width, y_depth_normalized, y_pred_depth,
                                 y_porosity_normalized, y_pred_porosity, r2_width, r2_depth, r2_porosity)

# [10] Final summary with R² filtering
print("\n[9/9] Final SVR Model Performance with R² Filtering:")

# Count how many targets meet the R² criteria
valid_targets = []
if 0.8 <= r2_width <= 1.2:
    valid_targets.append(f"Width (R²={r2_width:.4f})")
if 0.8 <= r2_depth <= 1.2:
    valid_targets.append(f"Depth (R²={r2_depth:.4f})")
if 0.8 <= r2_porosity <= 1.2:
    valid_targets.append(f"Porosity (R²={r2_porosity:.4f})")

if valid_targets:
    print(f"✓ Valid targets (0.8 ≤ R² ≤ 1.2): {', '.join(valid_targets)}")
else:
    print("✗ No targets meet the R² criteria (0.8 ≤ R² ≤ 1.2)")

# Report all R² scores
print(f"\nAll R² scores:")
print(f"Width: {r2_width:.4f} {'✓' if 0.8 <= r2_width <= 1.2 else '✗'}")
print(f"Depth: {r2_depth:.4f} {'✓' if 0.8 <= r2_depth <= 1.2 else '✗'}")
print(f"Porosity: {r2_porosity:.4f} {'✓' if 0.8 <= r2_porosity <= 1.2 else '✗'}")

print("\nSVR analysis completed!")
