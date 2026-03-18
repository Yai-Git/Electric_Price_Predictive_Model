import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold

# =============================================================================
# PROFILE 1: The "Geometric" Profile (Best for KNN, SVM)
# Strategy: KNN Imputation -> Min-Max Scaling (numeric), OHE (categorical) 
#           -> Low Variance Filter
# =============================================================================
def build_geometric_pipeline(num_cols, cat_cols):
    """
    Builds a Scikit-Learn pipeline tailored for algorithms that compute distances 
    (KNN, SVM, K-Means).
    """
    
    # Numeric Pipeline: KNN Imputation -> MinMaxScaler
    num_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', MinMaxScaler())
    ])
    
    # Categorical Pipeline: Most frequent Imputer -> OneHotEncoder
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='passthrough')
    
    # Final Pipeline with Variance Threshold (remove flat features)
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selector', VarianceThreshold(threshold=0.01)) # filter out features with nearly zero variance
    ])
    
    return full_pipeline


# =============================================================================
# PROFILE 2: The "Statistical" Profile (Best for Linear/Logistic Regression)
# Strategy: Mean/Median Imputer -> StanardScaler -> Optional Power/Log 
#           -> OHE with drop='first'
# =============================================================================
def log_transform(X):
    # Log1p transforms handling 0s
    return np.log1p(np.abs(X)) * np.sign(X)

def build_statistical_pipeline(num_cols, cat_cols, apply_log=False):
    """
    Builds a Scikit-Learn pipeline tailored for linear models assuming 
    normal distributions and punishing multicollinearity.
    """
    
    numeric_steps = [
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ]
    
    if apply_log:
        numeric_steps.insert(1, ('log_transform', FunctionTransformer(log_transform, validate=False)))
    
    num_pipeline = Pipeline(numeric_steps)
    
    # Categorical: drop_first=True prevents perfect multicollinearity for linear models
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='passthrough')
    
    return Pipeline([
        ('preprocessor', preprocessor)
    ])


# =============================================================================
# PROFILE 3: The "Tree-Ready" Profile (Best for Random Forest, XGBoost)
# Strategy: Constant Imputation (-999) -> No Scaling -> Ordinal Encoding
# =============================================================================
def build_tree_pipeline(num_cols, cat_cols):
    """
    Builds a Scikit-Learn pipeline tailored for tree-based models which handle
    scale variance and missingness perfectly on their own.
    """
    
    # Numeric Pipeline: Constant Impute (Trees can learn -999 is explicit missingness), no scaling
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=-999))
    ])
    
    # Categorical Pipeline: Ordinal encoding (Trees split ordinal numbers well without blowing up dimension size)
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ], remainder='passthrough')
    
    return Pipeline([
        ('preprocessor', preprocessor)
    ])
