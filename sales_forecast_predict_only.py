import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FAST PREDICTION - Using Saved Models")
print("="*80)

# Check if models exist
if not os.path.exists('catboost_optuna.cbm'):
    print("\nERROR: catboost_optuna.cbm not found!")
    print("Please run sales_forecast_retrain.py first to train models.")
    exit(1)

if not os.path.exists('xgboost_optuna.pkl'):
    print("\nERROR: xgboost_optuna.pkl not found!")
    print("Please run sales_forecast_retrain.py first to train models.")
    exit(1)

if not os.path.exists('optuna_study.pkl'):
    print("\nERROR: optuna_study.pkl not found!")
    print("Please run sales_forecast_retrain.py first to train models.")
    exit(1)

# Load models and study
print("\nLoading saved models...")
final_catboost = CatBoostRegressor()
final_catboost.load_model('catboost_optuna.cbm')

with open('xgboost_optuna.pkl', 'rb') as f:
    final_xgboost = pickle.load(f)

with open('optuna_study.pkl', 'rb') as f:
    study = pickle.load(f)

# Get ensemble weight from study
best_params = study.best_params
ensemble_weight = best_params['ensemble_weight']
weight_cat = 1 - ensemble_weight
weight_xgb = ensemble_weight

print("Models loaded successfully")
print(f"  Ensemble weights - CatBoost: {weight_cat:.3f}, XGBoost: {weight_xgb:.3f}")
print(f"  Best Forecast Accuracy from training: {-study.best_value:.2%}")

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

# PREPARE MONTHLY DATA
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

print(f"Monthly records: {len(monthly_train)}")
print(f"Features: {len(monthly_features)}")

# PREDICT MONTHLY TOTALS
print("\n" + "="*80)
print("Predicting Monthly Totals")
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
print(f"\nHistorical growth 2023-2024: {historical_growth*100:.1f}%")
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
    print("\nUsing model's raw prediction.")
    final_total = monthly_predictions.sum()
    applied_growth = None
else:
    try:
        applied_growth = float(growth_input.replace(',', '.'))
        # Apply growth to 2024 total, not raw prediction
        final_total = total_2024 * (1 + applied_growth / 100)
        print(f"\nApplying {applied_growth:+.1f}% growth to 2024 total.")
        print(f"  2024 total: {total_2024:,.0f}")
        print(f"  2025 target: {final_total:,.0f}")
    except:
        print("\nInvalid input. Using model's raw prediction.")
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

# STAGE 2: PRODUCT DISTRIBUTION
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
    print("ORDER column not found, using Units (actual sales)")

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
print(f"Saved to: {output_file}")
print("="*80)

print("\nTop 10 products:")
product_summary = forecast_output.groupby('Map EAN')['Units'].sum().sort_values(ascending=False).head(10)
for product, total in product_summary.items():
    print(f"  {product}: {total:>10,}")

print("\n" + "="*80)
print("PREDICTION COMPLETED SUCCESSFULLY")
print("="*80)
print("\nTo retrain models with new data, run: sales_forecast_retrain.py")
