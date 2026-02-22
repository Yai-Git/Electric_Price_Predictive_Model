import pandas as pd
import numpy as np

def load_and_pivot_data(filepath, start_date='2021-10-01'):
    """
    Loads raw electricity price data and pivots it to wide format.
    
    Args:
        filepath (str): Path to the CSV file.
        start_date (str): Filter data from this date.
        
    Returns:
        pd.DataFrame: Wide-format dataframe with date_time index and columns for each region.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    # Filter date
    if start_date:
        df = df[df['date_time'] >= start_date]
        
    # Pivot to wide format
    df_simple = df[['date_time', 'REGIONID', 'RRP']].copy()
    df_wide = df_simple.pivot(index='date_time', columns='REGIONID', values='RRP')
    df_wide = df_wide.reset_index()
    df_wide = df_wide.sort_values('date_time').reset_index(drop=True)
    df_wide.columns.name = None
    
    print(f"Data loaded. Shape: {df_wide.shape}")
    return df_wide

def calculate_caps(df_wide, regions=None, iqr_multiplier=1.5):
    """
    Calculates Price Caps (Min/Max) using IQR method.
    
    Args:
        df_wide (pd.DataFrame): Wide format data.
        regions (list): List of regions to calculate caps for.
        iqr_multiplier (float): Multiplier for IQR (1.5 or 3.0).
        
    Returns:
        dict: min_caps and max_caps dictionaries.
    """
    if regions is None:
        regions = [col for col in df_wide.columns if col != 'date_time']
        
    min_caps = {}
    max_caps = {}
    
    print("\nCalculating Caps:")
    for region in regions:
        series = df_wide[region].dropna()
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        min_caps[region] = Q1 - (iqr_multiplier * IQR)
        max_caps[region] = Q3 + (iqr_multiplier * IQR)
        
        print(f"  {region}: Min={min_caps[region]:.2f}, Max={max_caps[region]:.2f}")
        
    return min_caps, max_caps

def apply_capping(df_wide, min_caps, max_caps, regions=None):
    """
    Clips the dataframe values based on provided caps.
    """
    df_capped = df_wide.copy()
    if regions is None:
        regions = [col for col in df_wide.columns if col != 'date_time']
        
    for region in regions:
        if region in min_caps and region in max_caps:
            df_capped[region] = df_capped[region].clip(lower=min_caps[region], upper=max_caps[region])
            
    return df_capped

def create_lagged_features(df_wide, target_region, n_lags=6, horizon=6):
    """
    Creates lag features and target variable.
    
    Args:
        df_wide (pd.DataFrame): Input dataframe.
        target_region (str): The region to predict (e.g., 'NSW1').
        n_lags (int): Number of lags to create for ALL regions.
        horizon (int): Prediction horizon steps.
        
    Returns:
        pd.DataFrame: Feature-engineered dataframe with 'target' column.
        list: List of feature column names.
    """
    df_feat = df_wide.copy()
    regions = [col for col in df_feat.columns if col != 'date_time']
    
    # 1. Create Lags
    for region in regions:
        for lag in range(1, n_lags + 1):
            df_feat[f'{region}_lag{lag}'] = df_feat[region].shift(lag)
            
    # 2. Time Features
    df_feat['hour'] = df_feat['date_time'].dt.hour
    df_feat['day_of_week'] = df_feat['date_time'].dt.dayofweek
    df_feat['month'] = df_feat['date_time'].dt.month
    
    # 3. Create Target (Shifted Future)
    df_feat['target'] = df_feat[target_region].shift(-horizon)
    
    # Drop NaNs created by shifting
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    # Define Feature Columns
    lag_cols = [col for col in df_feat.columns if '_lag' in col]
    time_cols = ['hour', 'day_of_week', 'month']
    feature_cols = lag_cols + time_cols
    
    return df_feat, feature_cols
