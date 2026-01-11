import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna
from optuna.samplers import TPESampler
import pickle
import warnings
import hashlib
import os
warnings.filterwarnings('ignore')

print("="*80)
print("RETRAINING WITH OPTUNA OPTIMIZATION")
print("="*80)
print("\n⚠️  This will retrain models from scratch with hyperparameter optimization")
print("    This may take 10-20 minutes depending on your hardware.")

# Ask for confirmation
response = input("\nDo you want to continue? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("\n❌ Retraining cancelled.")
    exit(0)

# Ask for number of trials
print("\n" + "="*80)
print("Optuna Configuration")
print("="*80)
print("Number of trials determines optimization quality vs. time:")
print("  - 20 trials:  ~5-10 minutes (quick)")
print("  - 50 trials:  ~10-20 minutes (balanced, default)")
print("  - 100 trials: ~20-40 minutes (thorough)")

n_trials_input = input("\nEnter number of trials (press Enter for default 50): ").strip()
if n_trials_input == '':
    n_trials = 50
else:
    try:
        n_trials = int(n_trials_input)
        if n_trials < 10:
            print("⚠️  Minimum 10 trials required. Setting to 10.")
            n_trials = 10
        elif n_trials > 200:
            print("⚠️  Maximum 200 trials recommended. Setting to 200.")
            n_trials = 200
    except:
        print("⚠️  Invalid input. Using default 50 trials.")
        n_trials = 50

print(f"\n✓ Will run {n_trials} trials")

# Load data
print("\n" + "="*80)
print("Loading Data")
print("="*80)

training_data = pd.read_excel('trainingnew.xlsx')
forecast_data = pd.read_excel('forecastNEW.xlsx')

training_data['date'] = pd.to_datetime(training_data['date'])
forecast_data['date'] = pd.to_datetime(forecast_data['date'])

print(f"Training: {training_data.shape}")
print(f"Forecast: {forecast_data.shape}")

# ============================================================================
# PREPARE MONTHLY DATA
# ============================================================================
print("\n" + "="*80)
print("Preparing Monthly Data")
print("="*80)

# Aggregate to monthly level
monthly_train = training_data.groupby(training_data['date']).agg({
    'Units': 'sum',
    'SR RATE': 'mean',
    'IsBayram': 'max',
    'isZam': 'max',
    'ZamOran': 'mean'
}).reset_index()

monthly_train['year'] = monthly_train['date'].dt.year
monthly_train['month'] = monthly_train['date'].dt.month
monthly_train['quarter'] = monthly_train['date'].dt.quarter

# Time features
monthly_train['month_sin'] = np.sin(2 * np.pi * monthly_train['month'] / 12)
monthly_train['month_cos'] = np.cos(2 * np.pi * monthly_train['month'] / 12)
monthly_train['year_month'] = monthly_train['year'] * 12 + monthly_train['month']

# Lag features
monthly_train = monthly_train.sort_values('date')
for lag in [1, 2, 3, 6, 12]:
    monthly_train[f'total_lag_{lag}'] = monthly_train['Units'].shift(lag)

# YoY
monthly_train['total_yoy'] = monthly_train['Units'].shift(12)
monthly_train['total_yoy_growth'] = (monthly_train['Units'] - monthly_train['total_yoy']) / (monthly_train['total_yoy'] + 1)

# Rolling stats
for window in [3, 6, 12]:
    monthly_train[f'total_rolling_mean_{window}'] = monthly_train['Units'].rolling(window=window, min_periods=1).mean().shift(1)
    monthly_train[f'total_rolling_std_{window}'] = monthly_train['Units'].rolling(window=window, min_periods=1).std().shift(1)

# Monthly averages (excluding bayram months to avoid feature leakage)
# This forces the model to learn bayram effect separately via IsBayram feature
non_bayram_data = monthly_train[monthly_train['IsBayram'] == 0].copy()
month_avg_non_bayram = non_bayram_data.groupby('month')['Units'].mean().to_dict()
monthly_train['avg_by_month'] = monthly_train['month'].map(month_avg_non_bayram)
monthly_train['avg_by_month'] = monthly_train['avg_by_month'].fillna(monthly_train['Units'].mean())

monthly_train['avg_by_year'] = monthly_train.groupby('year')['Units'].transform('mean')

# Define features
monthly_features = [
    'SR RATE', 'ZamOran', 'IsBayram', 'isZam', 'month', 'quarter',
    'month_sin', 'month_cos', 'year_month',
    'total_yoy', 'total_yoy_growth',
    'avg_by_month', 'avg_by_year'
]
monthly_features.extend([col for col in monthly_train.columns if 'lag' in col or 'rolling' in col])
monthly_features = [f for f in monthly_features if f in monthly_train.columns]

X_monthly = monthly_train[monthly_features].fillna(0)
y_monthly = monthly_train['Units']

print(f"Monthly records: {len(monthly_train)}")
print(f"Features: {len(monthly_features)}")

# ============================================================================
# OPTUNA OPTIMIZATION
# ============================================================================
print("\n" + "="*80)
print("STAGE 1: Optimizing Hyperparameters with Optuna")
print("="*80)

# Split: Use last 3 months for validation (product-level)
unique_months = monthly_train['date'].unique()
unique_months_sorted = sorted(unique_months)
val_months = unique_months_sorted[-3:]  # Last 3 months
train_months = unique_months_sorted[:-3]  # All except last 3

monthly_train_split = monthly_train[monthly_train['date'].isin(train_months)].copy()
monthly_val_split = monthly_train[monthly_train['date'].isin(val_months)].copy()

X_train = monthly_train_split[monthly_features].fillna(0)
y_train = monthly_train_split['Units']
X_val = monthly_val_split[monthly_features].fillna(0)
y_val = monthly_val_split['Units']

# Get product-level validation data for accuracy calculation
val_product_data = training_data[training_data['date'].isin(val_months)].copy()
print(f"Validation: {len(val_months)} months, {len(val_product_data)} product-month records")

def objective(trial):
    """Optuna objective function to optimize both models and ensemble weight"""
    
    # CatBoost hyperparameters
    catboost_params = {
        'iterations': trial.suggest_int('cat_iterations', 500, 2000, step=100),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1, 10),
        'random_seed': 42,
        'verbose': False
    }
    
    # XGBoost hyperparameters
    xgboost_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 300, 1500, step=100),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 7),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('xgb_gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.5, 2.0),
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    # Ensemble weight (0 = all CatBoost, 1 = all XGBoost)
    ensemble_weight = trial.suggest_float('ensemble_weight', 0.3, 0.7)
    
    # Train CatBoost
    cat_model = CatBoostRegressor(**catboost_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    # Train XGBoost
    xgb_model = XGBRegressor(**xgboost_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Predict monthly totals for validation months
    pred_cat = cat_model.predict(X_val)
    pred_xgb = xgb_model.predict(X_val)
    pred_ensemble = (1 - ensemble_weight) * pred_cat + ensemble_weight * pred_xgb
    pred_ensemble = np.maximum(pred_ensemble, 0)
    
    # Create monthly predictions dataframe
    monthly_preds = monthly_val_split[['date', 'month']].copy()
    monthly_preds['predicted_total'] = pred_ensemble
    
    # Calculate product shares from training data (excluding validation months)
    train_product_data = training_data[training_data['date'].isin(train_months)].copy()
    
    # Use last 3 months of training data for WEIGHTED shares
    train_unique_dates = sorted(train_product_data['date'].unique(), reverse=True)
    last_3_train_dates = train_unique_dates[:3]
    
    # Weighted average: newest=0.45, middle=0.35, oldest=0.20
    weights_val = [0.20, 0.35, 0.45]
    monthly_shares_val = []
    
    for i, date in enumerate(last_3_train_dates):
        month_data = train_product_data[train_product_data['date'] == date].copy()
        
        if 'order' in month_data.columns:
            month_total = month_data['order'].sum()
            product_month = month_data.groupby('Map EAN')['order'].sum()
        else:
            month_total = month_data['Units'].sum()
            product_month = month_data.groupby('Map EAN')['Units'].sum()
        
        if month_total > 0:
            product_share = (product_month / month_total) * weights_val[i]
            monthly_shares_val.append(product_share)
    
    # Sum weighted shares
    product_shares = pd.concat(monthly_shares_val, axis=1).sum(axis=1)
    product_shares = product_shares / product_shares.sum()
    
    # Distribute monthly predictions to products
    product_forecasts = []
    for _, row in monthly_preds.iterrows():
        month = row['month']
        total = row['predicted_total']
        
        for product, share in product_shares.items():
            product_forecasts.append({
                'date': row['date'],
                'Map EAN': product,
                'forecast': total * share
            })
    
    forecast_df = pd.DataFrame(product_forecasts)
    
    # Merge with actual product-level validation data
    val_comparison = val_product_data[['date', 'Map EAN', 'Units']].merge(
        forecast_df, on=['date', 'Map EAN'], how='left'
    )
    val_comparison['forecast'] = val_comparison['forecast'].fillna(0)
    
    # Calculate PRODUCT-LEVEL Forecast Accuracy
    total_absolute_error = np.sum(np.abs(val_comparison['Units'] - val_comparison['forecast']))
    total_actual = np.sum(val_comparison['Units'])
    
    if total_actual > 0:
        forecast_accuracy = 1 - (total_absolute_error / total_actual)
    else:
        forecast_accuracy = 0
    
    # Return negative accuracy because Optuna minimizes (we want to maximize accuracy)
    return -forecast_accuracy

# Create study with SQLite storage
print(f"\nStarting Optuna optimization with {n_trials} trials...")
print("This may take a while. Progress will be shown below.")
print("-" * 80)

# Generate unique study name based on training data
# This ensures new study when training data changes
data_hash = hashlib.md5(str(len(training_data)).encode()).hexdigest()[:8]
study_name = f'sales_forecast_opt_{data_hash}'

storage_name = "sqlite:///optuna_study.db"
print(f"\nUsing SQLite storage: {storage_name}")
print(f"Study name: {study_name}")
print("Trial history will be saved and can be resumed later.")

# Check if study exists
try:
    existing_study = optuna.load_study(
        study_name=study_name,
        storage=storage_name
    )
    print(f"Found existing study with {len(existing_study.trials)} trials.")
    print(f"Will continue optimization from trial {len(existing_study.trials) + 1}.\n")
except:
    print("Creating new study.\n")

study = optuna.create_study(
    study_name=study_name,
    direction='minimize',  # Minimize negative accuracy = Maximize accuracy
    sampler=TPESampler(seed=42),
    storage=storage_name,
    load_if_exists=True
)

# Optimize
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

print("\n" + "="*80)
print("Optimization Results")
print("="*80)

print(f"\nBest Forecast Accuracy: {-study.best_value:.2%}")
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# ============================================================================
# TRAIN FINAL MODELS WITH BEST PARAMETERS
# ============================================================================
print("\n" + "="*80)
print("Training Final Models with Optimized Parameters")
print("="*80)

best_params = study.best_params

# Extract CatBoost params
catboost_best_params = {
    'iterations': best_params['cat_iterations'],
    'learning_rate': best_params['cat_learning_rate'],
    'depth': best_params['cat_depth'],
    'l2_leaf_reg': best_params['cat_l2_leaf_reg'],
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

# Extract XGBoost params
xgboost_best_params = {
    'n_estimators': best_params['xgb_n_estimators'],
    'learning_rate': best_params['xgb_learning_rate'],
    'max_depth': best_params['xgb_max_depth'],
    'min_child_weight': best_params['xgb_min_child_weight'],
    'subsample': best_params['xgb_subsample'],
    'colsample_bytree': best_params['xgb_colsample_bytree'],
    'gamma': best_params['xgb_gamma'],
    'reg_alpha': best_params['xgb_reg_alpha'],
    'reg_lambda': best_params['xgb_reg_lambda'],
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# Ensemble weight
ensemble_weight = best_params['ensemble_weight']
weight_cat = 1 - ensemble_weight
weight_xgb = ensemble_weight

print(f"\nEnsemble weights:")
print(f"  CatBoost: {weight_cat:.3f}")
print(f"  XGBoost:  {weight_xgb:.3f}")

# Train final models
print("\nTraining CatBoost...")
final_catboost = CatBoostRegressor(**catboost_best_params)
final_catboost.fit(X_train, y_train, eval_set=(X_val, y_val))

print("Training XGBoost...")
final_xgboost = XGBRegressor(**xgboost_best_params)
final_xgboost.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("Feature Importance Analysis")
print("="*80)

# CatBoost feature importance
cat_importance = final_catboost.get_feature_importance()
cat_feature_names = monthly_features
cat_importance_df = pd.DataFrame({
    'feature': cat_feature_names,
    'importance': cat_importance
}).sort_values('importance', ascending=False)

print("\nCatBoost - Top 15 Features:")
for idx, row in cat_importance_df.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:>8.2f}")

# XGBoost feature importance
xgb_importance = final_xgboost.feature_importances_
xgb_importance_df = pd.DataFrame({
    'feature': cat_feature_names,
    'importance': xgb_importance
}).sort_values('importance', ascending=False)

print("\nXGBoost - Top 15 Features:")
for idx, row in xgb_importance_df.head(15).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:>8.4f}")

# Highlight IsBayram specifically
print("\n" + "-"*80)
print("IsBayram Feature Importance:")
print("-"*80)
cat_bayram = cat_importance_df[cat_importance_df['feature'] == 'IsBayram']
xgb_bayram = xgb_importance_df[xgb_importance_df['feature'] == 'IsBayram']

if not cat_bayram.empty:
    cat_rank = cat_importance_df.index.get_loc(cat_bayram.index[0]) + 1
    print(f"  CatBoost: {cat_bayram['importance'].values[0]:>8.2f} (Rank: {cat_rank}/{len(cat_feature_names)})")
else:
    print("  CatBoost: Not found")

if not xgb_bayram.empty:
    xgb_rank = xgb_importance_df.index.get_loc(xgb_bayram.index[0]) + 1
    print(f"  XGBoost:  {xgb_bayram['importance'].values[0]:>8.4f} (Rank: {xgb_rank}/{len(cat_feature_names)})")
else:
    print("  XGBoost:  Not found")

# Validation metrics - PRODUCT LEVEL
val_pred_cat = final_catboost.predict(X_val)
val_pred_xgb = final_xgboost.predict(X_val)
val_pred_ensemble = weight_cat * val_pred_cat + weight_xgb * val_pred_xgb
val_pred_ensemble = np.maximum(val_pred_ensemble, 0)

# Calculate product-level accuracy for each model
def calc_product_level_accuracy(monthly_preds, model_name=""):
    """Calculate accuracy at product level by distributing monthly predictions"""
    monthly_df = monthly_val_split[['date', 'month']].copy()
    monthly_df['predicted_total'] = monthly_preds
    
    # Use same WEIGHTED share calculation as in objective
    train_product_data = training_data[training_data['date'].isin(train_months)].copy()
    train_unique_dates = sorted(train_product_data['date'].unique(), reverse=True)
    last_3_train_dates = train_unique_dates[:3]
    
    weights_val = [0.20, 0.35, 0.45]
    monthly_shares_val = []
    
    for i, date in enumerate(last_3_train_dates):
        month_data = train_product_data[train_product_data['date'] == date].copy()
        
        if 'order' in month_data.columns:
            month_total = month_data['order'].sum()
            product_month = month_data.groupby('Map EAN')['order'].sum()
        else:
            month_total = month_data['Units'].sum()
            product_month = month_data.groupby('Map EAN')['Units'].sum()
        
        if month_total > 0:
            product_share = (product_month / month_total) * weights_val[i]
            monthly_shares_val.append(product_share)
    
    product_shares = pd.concat(monthly_shares_val, axis=1).sum(axis=1)
    product_shares = product_shares / product_shares.sum()
    
    # Distribute to products
    product_forecasts = []
    for _, row in monthly_df.iterrows():
        for product, share in product_shares.items():
            product_forecasts.append({
                'date': row['date'],
                'Map EAN': product,
                'forecast': row['predicted_total'] * share
            })
    
    forecast_df = pd.DataFrame(product_forecasts)
    val_comparison = val_product_data[['date', 'Map EAN', 'Units']].merge(
        forecast_df, on=['date', 'Map EAN'], how='left'
    )
    val_comparison['forecast'] = val_comparison['forecast'].fillna(0)
    
    total_abs_error = np.sum(np.abs(val_comparison['Units'] - val_comparison['forecast']))
    total_actual = np.sum(val_comparison['Units'])
    
    return 1 - (total_abs_error / total_actual) if total_actual > 0 else 0

acc_cat = calc_product_level_accuracy(val_pred_cat, "CatBoost")
acc_xgb = calc_product_level_accuracy(val_pred_xgb, "XGBoost")
acc_ensemble = calc_product_level_accuracy(val_pred_ensemble, "Ensemble")

print("\nValidation Performance (Product-Level Accuracy):")
print(f"  CatBoost  - Accuracy: {acc_cat:.2%}, RMSE: {np.sqrt(mean_squared_error(y_val, val_pred_cat)):,.0f}, MAE: {mean_absolute_error(y_val, val_pred_cat):,.0f}")
print(f"  XGBoost   - Accuracy: {acc_xgb:.2%}, RMSE: {np.sqrt(mean_squared_error(y_val, val_pred_xgb)):,.0f}, MAE: {mean_absolute_error(y_val, val_pred_xgb):,.0f}")
print(f"  Ensemble  - Accuracy: {acc_ensemble:.2%}, RMSE: {np.sqrt(mean_squared_error(y_val, val_pred_ensemble)):,.0f}, MAE: {mean_absolute_error(y_val, val_pred_ensemble):,.0f}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "="*80)
print("Saving Models")
print("="*80)

final_catboost.save_model('catboost_optuna.cbm')
with open('xgboost_optuna.pkl', 'wb') as f:
    pickle.dump(final_xgboost, f)
with open('optuna_study.pkl', 'wb') as f:
    pickle.dump(study, f)

print("\n✓ Models saved successfully:")
print("  - catboost_optuna.cbm")
print("  - xgboost_optuna.pkl")
print("  - optuna_study.pkl")

# ============================================================================
# GENERATE FORECAST
# ============================================================================
print("\n" + "="*80)
print("Generating Forecast with New Models")
print("="*80)

# Prepare forecast data
monthly_forecast = forecast_data.groupby(forecast_data['date']).agg({
    'SR RATE': 'mean',
    'IsBayram': 'max',
    'isZam': 'max',
    'ZamOran': 'mean'
}).reset_index()

monthly_forecast['year'] = monthly_forecast['date'].dt.year
monthly_forecast['month'] = monthly_forecast['date'].dt.month
monthly_forecast['quarter'] = monthly_forecast['date'].dt.quarter
monthly_forecast['month_sin'] = np.sin(2 * np.pi * monthly_forecast['month'] / 12)
monthly_forecast['month_cos'] = np.cos(2 * np.pi * monthly_forecast['month'] / 12)
monthly_forecast['year_month'] = monthly_forecast['year'] * 12 + monthly_forecast['month']

# Combine for lag calculation
combined_monthly = pd.concat([monthly_train, monthly_forecast], ignore_index=True)
combined_monthly = combined_monthly.sort_values('date')

for lag in [1, 2, 3, 6, 12]:
    combined_monthly[f'total_lag_{lag}'] = combined_monthly['Units'].shift(lag)

combined_monthly['total_yoy'] = combined_monthly['Units'].shift(12)
combined_monthly['total_yoy_growth'] = (combined_monthly['Units'] - combined_monthly['total_yoy']) / (combined_monthly['total_yoy'] + 1)

for window in [3, 6, 12]:
    combined_monthly[f'total_rolling_mean_{window}'] = combined_monthly['Units'].rolling(window=window, min_periods=1).mean().shift(1)
    combined_monthly[f'total_rolling_std_{window}'] = combined_monthly['Units'].rolling(window=window, min_periods=1).std().shift(1)

# Stats from training (use same non-bayram averages as training)
# This ensures consistency with how the model was trained
month_stats = pd.DataFrame(list(month_avg_non_bayram.items()), columns=['month', 'avg_by_month'])

year_stats = monthly_train.groupby('year')['Units'].mean().reset_index()
year_stats.columns = ['year', 'avg_by_year']

combined_monthly = combined_monthly.merge(month_stats, on='month', how='left', suffixes=('', '_new'))
combined_monthly = combined_monthly.merge(year_stats, on='year', how='left', suffixes=('', '_new'))

for col in combined_monthly.columns:
    if col.endswith('_new'):
        base_col = col[:-4]
        combined_monthly[base_col] = combined_monthly[col]
        combined_monthly = combined_monthly.drop(columns=[col])

monthly_forecast = combined_monthly[combined_monthly['Units'].isna()].copy()

# Predict
X_forecast = monthly_forecast[monthly_features].fillna(0)

pred_cat = final_catboost.predict(X_forecast)
pred_xgb = final_xgboost.predict(X_forecast)
monthly_predictions = weight_cat * pred_cat + weight_xgb * pred_xgb
monthly_predictions = np.maximum(monthly_predictions, 0)

# Show growth info and ask for adjustment
train_2023 = training_data[training_data['date'].dt.year == 2023]
train_2024 = training_data[training_data['date'].dt.year == 2024]
total_2023 = train_2023['Units'].sum()
total_2024 = train_2024['Units'].sum()
historical_growth = (total_2024 - total_2023) / total_2023

print(f"\nModel's raw prediction: {monthly_predictions.sum():,.0f}")
print(f"\nHistorical growth 2023→2024: {historical_growth*100:.1f}%")
print(f"2023 total: {total_2023:,.0f}")
print(f"2024 total: {total_2024:,.0f}")

print("\n" + "="*80)
print("Growth Rate Adjustment (Optional)")
print("="*80)
print("Apply growth rate to 2024 total to project 2025.")
print("Examples: 1.2 for +1.2%, -0.5 for -0.5%, or press Enter to use model prediction.\n")

growth_input = input("Enter growth rate % vs 2024 (or press Enter to use model): ").strip()

if growth_input == '':
    # Use model's raw prediction
    print("\n✓ Using model's raw prediction.")
    final_total = monthly_predictions.sum()
    applied_growth = None
else:
    try:
        applied_growth = float(growth_input.replace(',', '.'))
        # Apply growth to 2024 total, not raw prediction
        final_total = total_2024 * (1 + applied_growth / 100)
        print(f"\n✓ Applying {applied_growth:+.1f}% growth to 2024 total.")
        print(f"  2024 total: {total_2024:,.0f}")
        print(f"  2025 target: {final_total:,.0f}")
    except:
        print("\n⚠️  Invalid input. Using model's raw prediction.")
        final_total = monthly_predictions.sum()
        applied_growth = None

# Adjust predictions to match target total
adjustment_factor = final_total / monthly_predictions.sum()
monthly_predictions = monthly_predictions * adjustment_factor
monthly_forecast['predicted_total'] = monthly_predictions

print(f"\nFinal total: {monthly_predictions.sum():,.0f}")
if applied_growth is not None:
    print(f"Growth vs 2024: {applied_growth:+.1f}%")
else:
    model_vs_2024 = ((monthly_predictions.sum() - total_2024) / total_2024) * 100
    print(f"Model's implied growth vs 2024: {model_vs_2024:+.1f}%")

# ============================================================================
# STAGE 2: PRODUCT DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("STAGE 2: Distribute to Products")
print("="*80)

training_data['year'] = training_data['date'].dt.year
training_data['month'] = training_data['date'].dt.month

# Get last 3 months dynamically
unique_dates = training_data['date'].unique()
unique_dates = sorted(unique_dates, reverse=True)
last_3_unique_dates = unique_dates[:3]
last_3_months_data = training_data[training_data['date'].isin(last_3_unique_dates)].copy()
last_3_dates = sorted(last_3_months_data['date'].dt.strftime('%Y-%m').unique())
print(f"Using last 3 months for product shares: {', '.join(last_3_dates)}")
print(f"Total rows from last 3 months: {len(last_3_months_data)}")

# Get products from last 3 months (not just latest month)
products_latest = set(last_3_months_data['Map EAN'].unique())
products_forecast = set(forecast_data['Map EAN'].unique())
discontinued_products = products_latest - products_forecast

print(f"Discontinued products: {len(discontinued_products)}")

# Calculate WEIGHTED AVERAGE shares from last 3 months (weights: 0.45, 0.35, 0.20)
# Most recent month gets highest weight to capture trends
weights = [0.20, 0.35, 0.45]  # oldest to newest
monthly_shares = []

for i, date in enumerate(last_3_unique_dates):
    month_data = last_3_months_data[last_3_months_data['date'] == date].copy()
    
    if 'order' in month_data.columns:
        month_total = month_data['order'].sum()
        product_month = month_data.groupby('Map EAN')['order'].sum()
    else:
        month_total = month_data['Units'].sum()
        product_month = month_data.groupby('Map EAN')['Units'].sum()
    
    # Calculate share for this month
    if month_total > 0:
        product_share = (product_month / month_total) * weights[i]
        monthly_shares.append(product_share)

if 'order' in last_3_months_data.columns:
    print("Using ORDER column for product shares (true demand)")
else:
    print("⚠️  ORDER column not found, using Units (actual sales)")

print(f"Weighted average: oldest={weights[0]:.0%}, middle={weights[1]:.0%}, newest={weights[2]:.0%}")

# Sum weighted shares across all months
product_avg_share = pd.concat(monthly_shares, axis=1).sum(axis=1).reset_index()
product_avg_share.columns = ['Map EAN', 'avg_share']

# Normalize to ensure sum = 1.0
total_share = product_avg_share['avg_share'].sum()
if total_share > 0:
    product_avg_share['avg_share'] = product_avg_share['avg_share'] / total_share

# Apply to all months
all_months = range(1, 13)
product_month_share = []
for month in all_months:
    for _, row in product_avg_share.iterrows():
        product_month_share.append({
            'month': month,
            'Map EAN': row['Map EAN'],
            'share': row['avg_share']
        })

product_month_share = pd.DataFrame(product_month_share)

# Remove discontinued and redistribute
discontinued_share_by_month = product_month_share[
    product_month_share['Map EAN'].isin(discontinued_products)
].groupby('month')['share'].sum().reset_index()
discontinued_share_by_month.columns = ['month', 'discontinued_share']

product_month_share = product_month_share[
    ~product_month_share['Map EAN'].isin(discontinued_products)
].copy()

product_month_share = product_month_share.merge(discontinued_share_by_month, on='month', how='left')
product_month_share['discontinued_share'] = product_month_share['discontinued_share'].fillna(0)

continuing_share_by_month = product_month_share.groupby('month')['share'].sum().reset_index()
continuing_share_by_month.columns = ['month', 'continuing_share']

product_month_share = product_month_share.merge(continuing_share_by_month, on='month')
product_month_share['boost_factor'] = (1.0) / (1.0 - product_month_share['discontinued_share'])
product_month_share['adjusted_share'] = product_month_share['share'] * product_month_share['boost_factor']
product_month_share['share'] = product_month_share['adjusted_share']

# Distribute
forecast_output = forecast_data.copy()
forecast_output['month'] = forecast_output['date'].dt.month

forecast_output = forecast_output.merge(
    monthly_forecast[['date', 'predicted_total']],
    on='date',
    how='left'
)

forecast_output = forecast_output.merge(
    product_month_share[['month', 'Map EAN', 'share']],
    on=['month', 'Map EAN'],
    how='left'
)

def normalize_shares(group):
    if group['share'].isna().all():
        group['share'] = 1.0 / len(group)
    else:
        group['share'] = group['share'].fillna(0)
        total = group['share'].sum()
        if total > 0:
            group['share'] = group['share'] / total
        else:
            group['share'] = 1.0 / len(group)
    return group

forecast_output = forecast_output.groupby('date').apply(normalize_shares).reset_index(drop=True)

forecast_output['Units'] = forecast_output['predicted_total'] * forecast_output['share']
forecast_output['Units'] = np.round(forecast_output['Units']).astype(int)

print(f"Final total: {forecast_output['Units'].sum():,.0f}")

# Save
output_file = 'forecastNEW_optuna.xlsx'
output_cols = ['date', 'Map EAN', 'Units', 'SR RATE', 'IsBayram', 'isZam', 'ZamOran']
forecast_output[output_cols].to_excel(output_file, index=False)

print(f"\n" + "="*80)
print(f"✓ Saved to: {output_file}")
print("="*80)

print("\nTop 10 products:")
product_summary = forecast_output.groupby('Map EAN')['Units'].sum().sort_values(ascending=False).head(10)
for product, total in product_summary.items():
    print(f"  {product}: {total:>10,}")

print("\n" + "="*80)
print("RETRAINING COMPLETED SUCCESSFULLY")
print("="*80)
print("\nNext steps:")
print("  - Use sales_forecast_predict_only.py for fast predictions")
print("  - Run this script again when you want to retrain with new data")
