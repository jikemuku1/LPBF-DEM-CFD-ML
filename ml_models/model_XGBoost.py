import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from tqdm import tqdm  # 使用tqdm显示进度条

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

# [5] 使用简化的参数网格以加速运行
print("\n[5/9] Setting up XGBoost with enhanced features...")

# 使用简化的参数网格（您可以选择改回原来的15,552个组合）
param_grid = {
    "n_estimators": [100, 300, 500],  # 3个
    "max_depth": [3, 7, 9],  # 3个
    "learning_rate": [0.01, 0.1],  # 2个
    "subsample": [0.8, 1.0],  # 2个
    "colsample_bytree": [0.8, 1.0],  # 2个
    "gamma": [0, 0.1],  # 2个
    "reg_alpha": [0, 0.1],  # 2个
    "reg_lambda": [1, 2]  # 2个
}

# 计算参数组合总数
from itertools import product

param_keys = list(param_grid.keys())
param_combinations = list(product(*param_grid.values()))
total_combinations = len(param_combinations)

print(f"Total parameter combinations: {total_combinations:,}")
print(f"Cross-validation folds: 10")
print(f"Total fits: {total_combinations * 10:,}")

kf = KFold(n_splits=10, shuffle=True, random_state=42)


# [6] Training function with tqdm progress bar
def train_and_evaluate_xgb(X, y, param_grid, kf, target_name):
    """Train and evaluate XGBoost with GridSearchCV and tqdm progress"""
    print(f"Training XGBoost for {target_name}...")
    start_time = time.time()

    xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=1)  # 使用n_jobs=1以便顺序执行，便于进度显示

    # 计算参数组合总数
    from itertools import product
    param_keys = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))
    total_combinations = len(param_combinations)

    print(f"  Total parameter combinations: {total_combinations:,}")
    print(f"  Cross-validation folds: {kf.get_n_splits()}")
    print(f"  Total fits: {total_combinations * kf.get_n_splits():,}")
    print(f"  Training with progress bar...")

    # 使用tqdm包装的简单网格搜索
    best_score = float('inf')
    best_params = None

    # 创建参数组合列表
    param_combos = list(product(*param_grid.values()))

    # 使用tqdm显示进度
    for combo in tqdm(param_combos, desc=f"Grid search for {target_name}", unit="combo"):
        params = dict(zip(param_keys, combo))

        # 创建模型
        model = xgb.XGBRegressor(random_state=42, n_jobs=1, **params)

        # 计算交叉验证分数
        scores = []
        for train_idx, val_idx in kf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            scores.append(score)

        # 平均分数
        mean_score = np.mean(scores)

        # 更新最佳分数和参数
        if mean_score < best_score:
            best_score = mean_score
            best_params = params

    # 使用最佳参数训练最终模型
    best_xgb = xgb.XGBRegressor(random_state=42, n_jobs=1, **best_params)
    best_xgb.fit(X, y)
    training_time = time.time() - start_time

    # Get honest cross-validated predictions
    y_pred = cross_val_predict(best_xgb, X, y, cv=kf, n_jobs=1)

    # Calculate metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    # Handle MAPE calculation with zero values
    mask = y != 0
    if mask.sum() > 0:
        mape = mean_absolute_percentage_error(y[mask], y_pred[mask])
    else:
        mape = float('inf')

    print(f"  ✓ XGBoost - {target_name}: R² = {r2:.4f}, MSE = {mse:.4e}, Time = {training_time:.2f}s")
    print(f"  Best parameters: {best_params}")

    return best_xgb, y_pred, mse, r2, mape, training_time


# [7] Train models with progress bars
print("\n[6/9] Training XGBoost models with engineered features...")
print("Training progress (3 models total):")
print("=" * 50)

# 使用tqdm显示总体进度
targets = ["Width", "Depth", "Porosity"]

# 训练Width模型
print(f"\n[1/3] Training for Width...")
best_xgb_width, y_pred_width, mse_width, r2_width, mape_width, time_width = train_and_evaluate_xgb(
    X_normalized, y_width_normalized, param_grid, kf, "Width"
)

# 训练Depth模型
print(f"\n[2/3] Training for Depth...")
best_xgb_depth, y_pred_depth, mse_depth, r2_depth, mape_depth, time_depth = train_and_evaluate_xgb(
    X_normalized, y_depth_normalized, param_grid, kf, "Depth"
)

# 训练Porosity模型
print(f"\n[3/3] Training for Porosity...")
best_xgb_porosity, y_pred_porosity, mse_porosity, r2_porosity, mape_porosity, time_porosity = train_and_evaluate_xgb(
    X_normalized, y_porosity_normalized, param_grid, kf, "Porosity"
)

print("\n✓ All 3 models trained successfully!")
print("=" * 50)

# [8] Results summary
print("\n[7/9] XGBoost Results Summary:")
print(f"MSE for Max Melt Pool Width: {mse_width:.4e}")
print(f"MSE for Max Melt Pool Depth: {mse_depth:.4e}")
print(f"MSE for Porosity: {mse_porosity:.4e}")

print(f"R² for Max Melt Pool Width: {r2_width:.4f}")
print(f"R² for Max Melt Pool Depth: {r2_depth:.4f}")
print(f"R² for Porosity: {r2_porosity:.4f}")

# [9] Feature importance analysis
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
importance_width = analyze_feature_importance(X_normalized, y_width_normalized, feature_names, best_xgb_width, "Width")
importance_depth = analyze_feature_importance(X_normalized, y_depth_normalized, feature_names, best_xgb_depth, "Depth")
importance_porosity = analyze_feature_importance(X_normalized, y_porosity_normalized, feature_names, best_xgb_porosity,
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

    plt.plot([0, 1], [0, 1], 'r--', color='black', linewidth=3.5, label='Ideal y=x')

    plt.xlabel('True Values (Normalized)', fontsize=20, fontweight='bold')
    plt.ylabel('Predicted Values (Normalized)', fontsize=20, fontweight='bold')
    plt.title('XGBoost Cross-Validation Results (Normalized)', fontsize=20, fontweight='bold')

    plt.xticks(fontsize=20, fontweight='bold')
    plt.yticks(fontsize=20, fontweight='bold')

    plt.legend(prop={'size': 15, 'weight': 'bold'})
    plt.grid(True, alpha=0.3)
    plt.savefig("Fig13e_XGBoost_CrossVal_Results.png", dpi=300, bbox_inches="tight")
plt.close()


# Generate plot with R² filtering
plot_combined_results_normalized(y_width_normalized, y_pred_width, y_depth_normalized, y_pred_depth,
                                 y_porosity_normalized, y_pred_porosity, r2_width, r2_depth, r2_porosity)

# [11] XGBoost specific analysis
print("\n[10/10] XGBoost Specific Analysis:")


def analyze_xgb_features(model, feature_names, target_name):
    """Analyze XGBoost specific features"""
    # Get feature importance from the model
    importance_scores = model.feature_importances_

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)

    print(f"\nTop 10 feature importances for {target_name} (XGBoost built-in):")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    # Get model parameters
    print(f"\nXGBoost parameters for {target_name}:")
    print(f"  Best n_estimators: {model.n_estimators}")
    print(f"  Best max_depth: {model.max_depth}")
    print(f"  Best learning_rate: {model.learning_rate}")
    print(f"  Best subsample: {model.subsample}")
    print(f"  Best colsample_bytree: {model.colsample_bytree}")

    return importance_df


print("XGBoost feature importance analysis:")
xgb_importance_width = analyze_xgb_features(best_xgb_width, feature_names, "Width")
xgb_importance_depth = analyze_xgb_features(best_xgb_depth, feature_names, "Depth")
xgb_importance_porosity = analyze_xgb_features(best_xgb_porosity, feature_names, "Porosity")

# [12] Final summary with R² filtering
print("\n[11/11] Final XGBoost Model Performance with R² Filtering:")

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
print(f"Width: {mse_width:.4e}, Depth: {mse_depth:.4e}, Porosity: {mse_porosity:.4e}")

print("\n" + "=" * 80)
print("XGBOOST REGRESSION ANALYSIS COMPLETED!")
