import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.neural_network import MLPRegressor
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

# [5] Define BPNN parameters (using BPNN training process)
print("\n[5/9] Setting up BPNN with enhanced features...")

# BPNN parameter grid (similar to your previous BPNN code)
param_grid = {
    'hidden_layer_sizes': [(50, 50), (100, 50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam', 'sgd'],
    'alpha': [6.5e-5],
    'learning_rate': ['constant'],
    'learning_rate_init': [0.002]
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)


# [6] Training function (using BPNN instead of Random Forest)
def train_and_evaluate_bpnn(X, y, param_grid, kf, target_name):
    """Train and evaluate BPNN with honest cross-validation"""
    print(f"Training BPNN for {target_name}...")
    start_time = time.time()

    # Use MLPRegressor (BPNN) instead of RandomForestRegressor
    bpnn = MLPRegressor(max_iter=500, random_state=42)
    grid_search = GridSearchCV(
        estimator=bpnn,
        param_grid=param_grid,
        cv=kf,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )

    grid_search.fit(X, y)
    best_bpnn = grid_search.best_estimator_
    training_time = time.time() - start_time

    # Get honest cross-validated predictions
    y_pred = cross_val_predict(best_bpnn, X, y, cv=kf, n_jobs=-1)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Handle MAPE calculation with zero values
    mask = y != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y[mask], y_pred[mask])
    else:
        mape = float('inf')

    print(f"  ✓ BPNN - {target_name}: R² = {r2:.4f}, MSE = {mse:.4e}, Time = {training_time:.2f}s")
    print(f"  Best parameters: {grid_search.best_params_}")

    return best_bpnn, y_pred, mse, r2, mape, training_time


# [7] Train models
print("\n[6/9] Training BPNN models with engineered features...")

best_bpnn_width, y_pred_width, mse_width, r2_width, mape_width, time_width = train_and_evaluate_bpnn(
    X_normalized, y_width_normalized, param_grid, kf, "Width"
)

best_bpnn_depth, y_pred_depth, mse_depth, r2_depth, mape_depth, time_depth = train_and_evaluate_bpnn(
    X_normalized, y_depth_normalized, param_grid, kf, "Depth"
)

best_bpnn_porosity, y_pred_porosity, mse_porosity, r2_porosity, mape_porosity, time_porosity = train_and_evaluate_bpnn(
    X_normalized, y_porosity_normalized, param_grid, kf, "Porosity"
)

# [8] Results summary
print("\n[7/9] BPNN Results Summary:")
print(f"MSE for Max Melt Pool Width: {mse_width:.4e}")
print(f"MSE for Max Melt Pool Depth: {mse_depth:.4e}")
print(f"MSE for Porosity: {mse_porosity:.4e}")

print(f"R² for Max Melt Pool Width: {r2_width:.4f}")
print(f"R² for Max Melt Pool Depth: {r2_depth:.4f}")
print(f"R² for Porosity: {r2_porosity:.4f}")

# [9] Feature importance analysis (using permutation importance)
print("\n[8/9] Analyzing feature importance...")


def analyze_feature_importance(X, y, feature_names, model, target_name):
    """Analyze feature importance using permutation importance"""
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_scores = perm_importance.importances_mean
    importance_std = perm_importance.importances_std

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores,
        'Std': importance_std
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 most important features for {target_name}:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")

    return importance_df


feature_names = feature_columns
importance_width = analyze_feature_importance(X_normalized, y_width_normalized, feature_names, best_bpnn_width, "Width")
importance_depth = analyze_feature_importance(X_normalized, y_depth_normalized, feature_names, best_bpnn_depth, "Depth")
importance_porosity = analyze_feature_importance(X_normalized, y_porosity_normalized, feature_names, best_bpnn_porosity,
                                                 "Porosity")

# [10] Plotting with R² filtering (0.8 ≤ R² ≤ 1.2)
print("\n[9/9] Generating plots with R² filtering (0.8 ≤ R² ≤ 1.2)...")


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
    plt.title('BPNN Cross-Validation Results (Normalized)', fontsize=20, fontweight='bold')

    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')
    plt.ylim(-0.1, 1.05)
    # plt.xlim(-0.05, 1.5)
    plt.legend(prop={'size': 15, 'weight': 'bold'})
    # plt.grid(True, alpha=0.3)
    plt.savefig("Fig13c_BPNN_CrossVal_Results.png", dpi=300, bbox_inches="tight")
plt.close()


# Generate plot with R² filtering
plot_combined_results_normalized(y_width_normalized, y_pred_width, y_depth_normalized, y_pred_depth,
                                 y_porosity_normalized, y_pred_porosity, r2_width, r2_depth, r2_porosity)

# [11] Final summary with R² filtering
print("\n[10/10] Final BPNN Model Performance with R² Filtering:")

# Count how many targets meet the R² criteria
valid_targets = []
if 0.8 <= r2_width <= 1.2:
    valid_targets.append(f"Width (R²={r2_width:.4f})")
if 0.8 <= r2_depth <= 0.95:
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
print(f"Depth: {r2_depth:.4f} {'✓' if 0.8 <= r2_depth <= 0.95 else '✗'}")
print(f"Porosity: {r2_porosity:.4f} {'✓' if 0.8 <= r2_porosity <= 1.2 else '✗'}")

# Performance assessment
all_r2_above_08 = all([r2_width >= 0.8, r2_depth >= 0.8, r2_porosity >= 0.8])

if all_r2_above_08:
    print("✓ EXCELLENT: All R² scores are above 0.8!")
else:
    models_above_08 = []
    if r2_width >= 0.8: models_above_08.append("Width")
    if r2_depth >= 0.8: models_above_08.append("Depth")
    if r2_porosity >= 0.8: models_above_08.append("Porosity")

    if models_above_08:
        print(f"✓ GOOD: {', '.join(models_above_08)} achieved R² > 0.8")

    models_below_08 = []
    if r2_width < 0.8: models_below_08.append(f"Width (R²={r2_width:.4f})")
    if r2_depth < 0.8: models_below_08.append(f"Depth (R²={r2_depth:.4f})")
    if r2_porosity < 0.8: models_below_08.append(f"Porosity (R²={r2_porosity:.4f})")

    if models_below_08:
        print(f"⚠️  Needs improvement: {', '.join(models_below_08)}")

print(f"\nTraining times:")
print(f"Width: {time_width:.2f}s, Depth: {time_depth:.2f}s, Porosity: {time_porosity:.2f}s")
print(f"Average training time: {(time_width + time_depth + time_porosity) / 3:.2f}s")

print("\nBPNN analysis completed!")

# [12] Additional BPNN-specific analysis
print("\n[11/11] BPNN Specific Analysis:")


def analyze_bpnn_features(model, feature_names, target_name):
    """Analyze BPNN specific features"""
    # For BPNN, we can analyze weights and biases
    if hasattr(model, 'coefs_'):
        print(f"\nBPNN architecture for {target_name}:")
        for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
            print(f"  Layer {i + 1}: Weights shape {coef.shape}, Biases shape {intercept.shape}")

    # Get feature importance from permutation importance
    perm_importance = permutation_importance(
        model, X_normalized,
        y_width_normalized if target_name == "Width" else
        y_depth_normalized if target_name == "Depth" else y_porosity_normalized,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_scores = perm_importance.importances_mean
    importance_std = perm_importance.importances_std

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores,
        'Std': importance_std
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 feature importances for {target_name}:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")

    return importance_df


print("BPNN feature analysis:")
bpnn_importance_width = analyze_bpnn_features(best_bpnn_width, feature_names, "Width")
bpnn_importance_depth = analyze_bpnn_features(best_bpnn_depth, feature_names, "Depth")
bpnn_importance_porosity = analyze_bpnn_features(best_bpnn_porosity, feature_names, "Porosity")

print("\n" + "=" * 80)
print("BPNN REGRESSION ANALYSIS COMPLETED!")
print("=" * 80)


