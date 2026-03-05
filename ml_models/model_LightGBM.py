import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import lightgbm as lgb
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from itertools import product
from tqdm import tqdm

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

# [2] Feature engineering
print("\n[2/9] Creating comprehensive engineered features...")

def create_comprehensive_features(df):
    print("Creating comprehensive features with radius engineering...")
    power = df["Power [W]"].values
    velocity = df["Velocity [mm s-1]"].values
    radius = df["Radius [mm]"].values

    df['Linear_Energy_Density'] = power / velocity
    df['Area_Energy_Density'] = power / (velocity * radius)
    df['Volumetric_Energy_Density'] = power / (velocity * radius * 0.1)
    df['Specific_Energy'] = power / (velocity * radius ** 2)
    df['Energy_Flux'] = power / (np.pi * radius ** 2)
    df['Beam_Area'] = np.pi * (radius ** 2)
    df['Beam_Circumference'] = 2 * np.pi * radius
    df['Beam_Diameter'] = 2 * radius
    df['Aspect_Ratio_Energy'] = power / (velocity * radius)
    df['Power_per_Area'] = power / (np.pi * radius ** 2)
    df['Power_per_Radius'] = power / radius
    df['Intensity'] = power / (np.pi * (radius ** 2))
    df['Specific_Power'] = power / radius
    df['Stability_Index'] = (power * radius) / velocity
    df['Process_Efficiency'] = power / (velocity * radius ** 3)
    df['Thermal_Gradient'] = power / (radius ** 2)
    df['Melting_Parameter'] = power * radius / velocity
    df['Heat_Affected_Zone'] = radius ** 2 / velocity
    df['Radius_Velocity_Ratio'] = radius / velocity
    df['Power_Radius_Ratio'] = power / radius
    df['Energy_Concentration'] = power / (radius ** 2)
    df['Dimensionless_Energy'] = (power * radius) / (velocity ** 2)
    df['Power_Radius_Interaction'] = power * radius
    df['Velocity_Radius_Interaction'] = velocity * radius
    df['Power_Velocity_Radius'] = power * velocity * radius
    df['P_V_R_Combined'] = power / (velocity * radius)
    df['Log_Power'] = np.log(power + 1e-6)
    df['Log_Velocity'] = np.log(velocity + 1e-6)
    df['Log_Radius'] = np.log(radius + 1e-6)
    df['Radius_Squared'] = radius ** 2
    df['Inverse_Radius'] = 1 / (radius + 1e-6)
    df['Energy_Density_Radius'] = power / (velocity * radius ** 3)
    df['Conduction_Parameter'] = power / (velocity * radius)
    df['Penetration_Ratio'] = power / (velocity * radius ** 0.5)
    df['Fusion_Parameter'] = (power * radius) / velocity

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    total_features = len([col for col in df.columns if col not in
                          ['Power [W]', 'Velocity [mm s-1]', 'Radius [mm]',
                           'Max Melt Pool Width [mm]', 'Max Melt Pool Depth [mm]', 'Porosity [%]']])
    print(f"✓ Created {total_features} engineered features")
    return df

df_engineered = create_comprehensive_features(df)

feature_columns = [col for col in df_engineered.columns if col not in
                   ['Max Melt Pool Width [mm]', 'Max Melt Pool Depth [mm]', 'Porosity [%]']]

X_raw = df_engineered[feature_columns]
y_width = df_engineered["Max Melt Pool Width [mm]"]
y_depth = df_engineered["Max Melt Pool Depth [mm]"]
y_porosity = df_engineered["Porosity [%]"]

print(f"✓ Engineered dataset shape: {df_engineered.shape}")
print(f"✓ Number of features: {len(feature_columns)}")

# [3] Outlier removal
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

final_mask = mask_w & mask_d & mask_p
print(f"→ Final dataset after outlier removal: {final_mask.sum()} / {len(df_engineered)} samples")

X = X_raw[final_mask].reset_index(drop=True)
y_width = y_width[final_mask].reset_index(drop=True)
y_depth = y_depth[final_mask].reset_index(drop=True)
y_porosity = y_porosity[final_mask].reset_index(drop=True)

# [4] Normalization
print("\n[4/9] Preparing normalized data...")

scaler_X = MinMaxScaler()
scaler_y_width = MinMaxScaler()
scaler_y_depth = MinMaxScaler()
scaler_y_porosity = MinMaxScaler()

X_normalized = scaler_X.fit_transform(X)
y_width_normalized = scaler_y_width.fit_transform(y_width.values.reshape(-1, 1)).ravel()
y_depth_normalized = scaler_y_depth.fit_transform(y_depth.values.reshape(-1, 1)).ravel()
y_porosity_normalized = scaler_y_porosity.fit_transform(y_porosity.values.reshape(-1, 1)).ravel()

# [5] LightGBM parameter grid (practical size)
print("\n[5/9] Setting up LightGBM parameters...")

param_grid_lgb = {
    'n_estimators': [200, 500],
    'max_depth': [5, 7, -1],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 63],
    'subsample': [0.9, 1.0],
    'colsample_bytree': [0.9, 1.0],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 2],
    'min_child_samples': [20]
}

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# [6] Training function with progress bar
def train_and_evaluate_lgb(X, y, param_grid, kf, target_name):
    print(f"Training LightGBM for {target_name}...")
    start_time = time.time()

    param_keys = list(param_grid.keys())
    param_combos = list(product(*param_grid.values()))

    best_score = float('inf')
    best_params = None

    with tqdm(total=len(param_combos), desc=f"Grid search {target_name}", unit="combo") as pbar:
        for combo in param_combos:
            params = dict(zip(param_keys, combo))
            model = lgb.LGBMRegressor(
                **params,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )

            scores = cross_validate(
                model, X, y,
                cv=kf,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            mean_score = -np.mean(scores['test_score'])

            if mean_score < best_score:
                best_score = mean_score
                best_params = params

            pbar.update(1)
            pbar.set_postfix(best_mse=f"{best_score:.2e}")

    best_lgb = lgb.LGBMRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    best_lgb.fit(X, y)

    training_time = time.time() - start_time

    y_pred = cross_val_predict(best_lgb, X, y, cv=kf, n_jobs=-1)

    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    mask = y != 0
    mape = mean_absolute_percentage_error(y[mask], y_pred[mask]) if mask.sum() > 0 else float('inf')

    print(f"  ✓ LightGBM - {target_name}: R² = {r2:.4f}, MSE = {mse:.4e}, Time = {training_time:.2f}s")
    print(f"  Best parameters: {best_params}")

    return best_lgb, y_pred, mse, r2, mape, training_time

# [7] Train models
print("\n[6/9] Training LightGBM models with engineered features...")

best_lgb_width, y_pred_width, mse_width, r2_width, mape_width, time_width = train_and_evaluate_lgb(
    X_normalized, y_width_normalized, param_grid_lgb, kf, "Width"
)

best_lgb_depth, y_pred_depth, mse_depth, r2_depth, mape_depth, time_depth = train_and_evaluate_lgb(
    X_normalized, y_depth_normalized, param_grid_lgb, kf, "Depth"
)

best_lgb_porosity, y_pred_porosity, mse_porosity, r2_porosity, mape_porosity, time_porosity = train_and_evaluate_lgb(
    X_normalized, y_porosity_normalized, param_grid_lgb, kf, "Porosity"
)

# [8] Results summary
print("\n[7/9] LightGBM Results Summary:")
print(f"MSE for Max Melt Pool Width: {mse_width:.4e}")
print(f"MSE for Max Melt Pool Depth: {mse_depth:.4e}")
print(f"MSE for Porosity: {mse_porosity:.4e}")

print(f"R² for Max Melt Pool Width: {r2_width:.4f}")
print(f"R² for Max Melt Pool Depth: {r2_depth:.4f}")
print(f"R² for Porosity: {r2_porosity:.4f}")

# Diagnostic print for porosity
print(f"\n→ Porosity R² = {r2_porosity:.4f} (will be shown regardless of value)")

# [9] Feature importance (permutation)
print("\n[8/9] Analyzing feature importance (permutation)...")

def analyze_feature_importance(X, y, feature_names, model, target_name):
    perm_importance = permutation_importance(
        model, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': perm_importance.importances_mean,
        'Std': perm_importance.importances_std
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 most important features for {target_name} (permutation):")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")
    return importance_df

feature_names = feature_columns
importance_width = analyze_feature_importance(X_normalized, y_width_normalized, feature_names, best_lgb_width, "Width")
importance_depth = analyze_feature_importance(X_normalized, y_depth_normalized, feature_names, best_lgb_depth, "Depth")
importance_porosity = analyze_feature_importance(X_normalized, y_porosity_normalized, feature_names, best_lgb_porosity, "Porosity")

# [10] Plotting — Porosity is now always shown
print("\n[9/9] Generating plots with R² filtering (Width & Depth filtered, Porosity always shown)...")

def plot_combined_results_normalized(y_true_width, y_pred_width, y_true_depth, y_pred_depth,
                                    y_true_porosity, y_pred_porosity, r2_width, r2_depth,
                                    r2_porosity):
    plt.figure(figsize=(10, 8))

    # Width — filtered
    plt.scatter(y_true_width, y_pred_width, marker='o', s=200, color='lightblue',
                edgecolor="black", linewidths=1.5, alpha=0.9,
                label=f'Molten Pool Width (R²: {r2_width:.4f})')

    # Depth — filtered
    plt.scatter(y_true_depth, y_pred_depth, marker='s', s=200, color='lightgreen',
                edgecolor="black", linewidths=1.5, alpha=0.9,
                label=f'Molten Pool Depth (R²: {r2_depth:.4f})')

    # Porosity — always shown, no if condition
    plt.scatter(y_true_porosity, y_pred_porosity, marker='^', s=200, color='orange',
                edgecolor="black", linewidths=1.5, alpha=0.9,
                label=f'Porosity (R²: {r2_porosity:.4f})')

    # Red dashed ideal line
    plt.plot([0, 1], [0, 1], 'r--', linewidth=3.5, label='Ideal y=x')

    # plt.xlabel('True Values (Normalized)', fontsize=20, fontweight='bold')
    # plt.ylabel('Predicted Values (Normalized)', fontsize=20, fontweight='bold')
    plt.xlabel('True Values (Normalized)', fontsize=23, fontweight='bold')
    plt.ylabel('Predicted Values (Normalized)', fontsize=23, fontweight='bold')
    plt.title('LightGBM Cross-Validation Results (Normalized)', fontsize=20, fontweight='bold')


    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    plt.legend(prop={'size': 16, 'weight': 'bold'}, loc='upper left',framealpha=1)

    # plt.grid(True, alpha=0.3, linestyle='--')
    plt.grid(False, alpha=0, linestyle='--')

    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig("Fig13g_LightGBM_CrossVal_Results.png", dpi=300, bbox_inches="tight")
plt.close()

plot_combined_results_normalized(y_width_normalized, y_pred_width, y_depth_normalized, y_pred_depth,
                                 y_porosity_normalized, y_pred_porosity, r2_width, r2_depth, r2_porosity)

# [11] LightGBM specific analysis
print("\n[10/10] LightGBM Specific Analysis:")

def analyze_lgb_features(model, feature_names, target_name):
    importance_scores = model.feature_importances_
    total = importance_scores.sum()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores / total if total > 0 else importance_scores
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 feature importances for {target_name} (LightGBM built-in):")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    print(f"\nLightGBM parameters for {target_name}:")
    print(f"  Best n_estimators: {model.n_estimators_}")
    print(f"  Best max_depth: {model.max_depth_}")
    print(f"  Best learning_rate: {model.learning_rate}")
    print(f"  Best num_leaves: {model.num_leaves_}")
    print(f"  Best colsample_bytree: {model.colsample_bytree_}")

    return importance_df

print("LightGBM feature importance analysis:")
lgb_importance_width = analyze_lgb_features(best_lgb_width, feature_names, "Width")
lgb_importance_depth = analyze_lgb_features(best_lgb_depth, feature_names, "Depth")
lgb_importance_porosity = analyze_lgb_features(best_lgb_porosity, feature_names, "Porosity")

# [12] Final summary
print("\n[11/11] Final LightGBM Model Performance with R² Filtering:")

valid_targets = []
if 0.8 <= r2_width <= 1.2: valid_targets.append(f"Width (R²={r2_width:.4f})")
if 0.8 <= r2_depth <= 1.2: valid_targets.append(f"Depth (R²={r2_depth:.4f})")
if 0.8 <= r2_porosity <= 1.2: valid_targets.append(f"Porosity (R²={r2_porosity:.4f})")

if valid_targets:
    print(f"✓ Valid targets (0.8 ≤ R² ≤ 1.2): {', '.join(valid_targets)}")
else:
    print("✗ No targets meet the R² criteria (0.8 ≤ R² ≤ 1.2)")

print(f"\nAll R² scores:")
print(f"Width: {r2_width:.4f} {'✓' if 0.8 <= r2_width <= 1.2 else '✗'}")
print(f"Depth: {r2_depth:.4f} {'✓' if 0.8 <= r2_depth <= 1.2 else '✗'}")
print(f"Porosity: {r2_porosity:.4f} {'✓' if 0.8 <= r2_porosity <= 1.2 else '✗ (shown anyway)'}")

print(f"\nTraining times:")
print(f"Width: {time_width:.2f}s, Depth: {time_depth:.2f}s, Porosity: {time_porosity:.2f}s")
print(f"Average training time: {(time_width + time_depth + time_porosity) / 3:.2f}s")

print("\n" + "=" * 80)
print("LIGHTGBM REGRESSION ANALYSIS COMPLETED!")
print("=" * 80)
