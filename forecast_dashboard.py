import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from catboost import CatBoostRegressor
import pickle
import os

# Page config
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Sales Forecast Dashboard")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    try:
        # Load forecast
        forecast = pd.read_excel('forecastNEW_optuna.xlsx')
        forecast['date'] = pd.to_datetime(forecast['date'])
        forecast.rename(columns={'Units': 'forecast'}, inplace=True)  # Rename for consistency
        
        # Load training data
        training = pd.read_excel('trainingnew.xlsx')
        training['date'] = pd.to_datetime(training['date'])
        
        return forecast, training
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load models and get feature importance
@st.cache_data
def load_feature_importance():
    try:
        # Load models
        catboost_model = CatBoostRegressor()
        catboost_model.load_model('catboost_optuna.cbm')
        
        with open('xgboost_optuna.pkl', 'rb') as f:
            xgboost_model = pickle.load(f)
        
        with open('optuna_study.pkl', 'rb') as f:
            study = pickle.load(f)
        
        # Get feature names
        feature_names = catboost_model.feature_names_
        
        # Get CatBoost feature importance
        cat_importance = catboost_model.get_feature_importance()
        
        # Get XGBoost feature importance
        xgb_importance = xgboost_model.feature_importances_
        
        # Get ensemble weight
        ensemble_weight = study.best_params['ensemble_weight']
        weight_cat = 1 - ensemble_weight
        weight_xgb = ensemble_weight
        
        # Calculate weighted importance
        weighted_importance = weight_cat * cat_importance + weight_xgb * xgb_importance
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'CatBoost': cat_importance,
            'XGBoost': xgb_importance,
            'Weighted': weighted_importance
        })
        
        # Sort by weighted importance
        importance_df = importance_df.sort_values('Weighted', ascending=False)
        
        # Get best accuracy
        best_accuracy = -study.best_value
        
        return importance_df, best_accuracy, weight_cat, weight_xgb
        
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
        return None, None, None, None

forecast, training = load_data()

if forecast is not None and training is not None:
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Product filter
    all_products = sorted(forecast['Map EAN'].unique())
    selected_products = st.sidebar.multiselect(
        "Select Products",
        options=all_products,
        default=all_products[:10]  # Top 10 by default
    )
    
    # Month filter
    forecast_months = sorted(forecast['date'].dt.strftime('%Y-%m').unique())
    selected_months = st.sidebar.multiselect(
        "Select Months",
        options=forecast_months,
        default=forecast_months
    )
    
    # Filter data
    if selected_products and selected_months:
        forecast_filtered = forecast[
            (forecast['Map EAN'].isin(selected_products)) &
            (forecast['date'].dt.strftime('%Y-%m').isin(selected_months))
        ]
    else:
        forecast_filtered = forecast
    
    # KPIs
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_forecast = forecast_filtered['forecast'].sum()
        st.metric("Total Forecast", f"{total_forecast:,.0f}")
    
    with col2:
        total_products = forecast_filtered['Map EAN'].nunique()
        st.metric("Products", f"{total_products}")
    
    with col3:
        avg_monthly = forecast_filtered.groupby('date')['forecast'].sum().mean()
        st.metric("Avg Monthly", f"{avg_monthly:,.0f}")
    
    with col4:
        # Calculate 2024 total from training
        training_2024 = training[training['date'].dt.year == 2024]
        total_2024 = training_2024['Units'].sum()
        growth = ((total_forecast - total_2024) / total_2024) * 100
        st.metric("Growth vs 2024", f"{growth:+.1f}%")
    
    st.markdown("---")
    
    # Monthly trend
    st.header("Monthly Forecast Trend")
    monthly_total = forecast_filtered.groupby('date')['forecast'].sum().reset_index()
    monthly_total['month'] = monthly_total['date'].dt.strftime('%Y-%m')
    
    fig_monthly = px.bar(
        monthly_total,
        x='month',
        y='forecast',
        title='Monthly Total Forecast',
        labels={'forecast': 'Forecast Units', 'month': 'Month'},
        color='forecast',
        color_continuous_scale='Blues'
    )
    fig_monthly.update_layout(height=400)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Product comparison
    st.header("Top Products")
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 15 products
        top_products = forecast_filtered.groupby('Map EAN')['forecast'].sum().nlargest(15).reset_index()
        
        fig_top = px.bar(
            top_products,
            x='forecast',
            y='Map EAN',
            orientation='h',
            title='Top 15 Products by Forecast',
            labels={'forecast': 'Total Forecast', 'Map EAN': 'Product'},
            color='forecast',
            color_continuous_scale='Viridis'
        )
        fig_top.update_layout(height=500)
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Product share pie chart
        top_10_share = forecast_filtered.groupby('Map EAN')['forecast'].sum().nlargest(10).reset_index()
        others = forecast_filtered[~forecast_filtered['Map EAN'].isin(top_10_share['Map EAN'])]['forecast'].sum()
        
        if others > 0:
            top_10_share = pd.concat([
                top_10_share,
                pd.DataFrame({'Map EAN': ['Others'], 'forecast': [others]})
            ])
        
        fig_pie = px.pie(
            top_10_share,
            values='forecast',
            names='Map EAN',
            title='Product Share (Top 10 + Others)',
            hole=0.4
        )
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Historical vs Forecast comparison
    st.header("Historical vs Forecast")
    
    # Prepare historical data (2024)
    hist_2024 = training[training['date'].dt.year == 2024].copy()
    hist_monthly = hist_2024.groupby(hist_2024['date'].dt.to_period('M'))['Units'].sum().reset_index()
    hist_monthly['date'] = hist_monthly['date'].dt.to_timestamp()
    hist_monthly['type'] = 'Historical (2024)'
    hist_monthly.rename(columns={'Units': 'value'}, inplace=True)
    
    # Prepare forecast data (2025)
    forecast_monthly = forecast.groupby('date')['forecast'].sum().reset_index()
    forecast_monthly['type'] = 'Forecast (2025)'
    forecast_monthly.rename(columns={'forecast': 'value'}, inplace=True)
    
    # Combine
    comparison = pd.concat([hist_monthly, forecast_monthly])
    comparison['month'] = comparison['date'].dt.strftime('%B')
    
    fig_comparison = px.line(
        comparison,
        x='month',
        y='value',
        color='type',
        title='2024 Historical vs 2025 Forecast (Monthly)',
        labels={'value': 'Units', 'month': 'Month'},
        markers=True
    )
    fig_comparison.update_layout(height=400)
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Product detail table
    st.header("Product Forecast Details")
    
    # Aggregate by product
    product_summary = forecast_filtered.groupby('Map EAN').agg({
        'forecast': ['sum', 'mean', 'std']
    }).reset_index()
    product_summary.columns = ['Product', 'Total Forecast', 'Avg Monthly', 'Std Dev']
    product_summary = product_summary.sort_values('Total Forecast', ascending=False)
    
    # Format numbers
    product_summary['Total Forecast'] = product_summary['Total Forecast'].apply(lambda x: f"{x:,.0f}")
    product_summary['Avg Monthly'] = product_summary['Avg Monthly'].apply(lambda x: f"{x:,.0f}")
    product_summary['Std Dev'] = product_summary['Std Dev'].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(product_summary, use_container_width=True, height=400)
    
    # Feature Importance Section
    st.header("Feature Importance Analysis")
    
    importance_df, best_accuracy, weight_cat, weight_xgb = load_feature_importance()
    
    if importance_df is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top 15 features
            top_features = importance_df.head(15).copy()
            
            fig_importance = go.Figure()
            
            # Add CatBoost importance
            fig_importance.add_trace(go.Bar(
                name='CatBoost',
                x=top_features['Feature'],
                y=top_features['CatBoost'],
                marker_color='lightblue'
            ))
            
            # Add XGBoost importance
            fig_importance.add_trace(go.Bar(
                name='XGBoost',
                x=top_features['Feature'],
                y=top_features['XGBoost'],
                marker_color='lightcoral'
            ))
            
            # Add Weighted importance
            fig_importance.add_trace(go.Bar(
                name='Weighted Ensemble',
                x=top_features['Feature'],
                y=top_features['Weighted'],
                marker_color='gold'
            ))
            
            fig_importance.update_layout(
                title='Top 15 Most Important Features',
                xaxis_title='Feature',
                yaxis_title='Importance Score',
                barmode='group',
                height=500,
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            # Model performance metrics
            st.subheader("Model Performance")
            
            if best_accuracy is not None:
                st.metric("Best Accuracy", f"{best_accuracy:.2%}")
            
            st.metric("CatBoost Weight", f"{weight_cat:.1%}")
            st.metric("XGBoost Weight", f"{weight_xgb:.1%}")
            
            st.markdown("---")
            
            # Feature categories
            st.subheader("Feature Categories")
            
            lag_features = importance_df[importance_df['Feature'].str.contains('lag', case=False)]
            rolling_features = importance_df[importance_df['Feature'].str.contains('rolling', case=False)]
            yoy_features = importance_df[importance_df['Feature'].str.contains('yoy', case=False)]
            
            st.write(f"**Lag Features:** {len(lag_features)}")
            st.write(f"**Rolling Features:** {len(rolling_features)}")
            st.write(f"**YoY Features:** {len(yoy_features)}")
            
            # Top 5 features
            st.markdown("---")
            st.subheader("Top 5 Features")
            for i, row in importance_df.head(5).iterrows():
                st.write(f"{i+1}. **{row['Feature']}**")
                st.progress(float(row['Weighted'] / importance_df['Weighted'].max()))
        
        # Detailed feature table
        st.subheader("All Features Importance")
        
        # Format the dataframe
        importance_display = importance_df.copy()
        importance_display['CatBoost'] = importance_display['CatBoost'].apply(lambda x: f"{x:.4f}")
        importance_display['XGBoost'] = importance_display['XGBoost'].apply(lambda x: f"{x:.4f}")
        importance_display['Weighted'] = importance_display['Weighted'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(importance_display, use_container_width=True, height=400)
    
    else:
        st.warning("Feature importance data not available. Please ensure models are trained.")
    
    # Download section
    st.header("Download Data")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download filtered forecast
        csv_forecast = forecast_filtered.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Forecast (CSV)",
            data=csv_forecast,
            file_name=f"forecast_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download product summary
        csv_summary = product_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Product Summary (CSV)",
            data=csv_summary,
            file_name=f"product_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.header("Model Info")
    
    # Load model performance
    importance_df, best_accuracy, weight_cat, weight_xgb = load_feature_importance()
    
    if best_accuracy is not None:
        st.sidebar.metric("Model Accuracy", f"{best_accuracy:.2%}")
        st.sidebar.metric("CatBoost Weight", f"{weight_cat:.1%}")
        st.sidebar.metric("XGBoost Weight", f"{weight_xgb:.1%}")
    
    st.sidebar.info("""
    **Forecast Model:**
    - Hybrid CatBoost + XGBoost
    - Optimized with Optuna
    - Product-level validation
    - Weighted share distribution (45%, 35%, 20%)
    - Uses ORDER data for true demand
    """)
    
else:
    st.error("Could not load forecast data. Please ensure 'forecastNEW_optuna.xlsx' exists.")
    st.info("Run `sales_forecast_retrain.py` or `sales_forecast_predict_only.py` first to generate forecast.")
