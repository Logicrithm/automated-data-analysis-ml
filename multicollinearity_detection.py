import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def detect_multicollinearity(X):
    """Return a DataFrame of VIF for each feature"""
    
    # 🔧 FIX 1: Handle NaN values - drop rows with any NaN
    X_clean = X.dropna()
    
    if X_clean.empty or len(X_clean) == 0:
        return pd.DataFrame({"Feature": X.columns, "VIF": [np.inf] * len(X.columns)})
    
    # 🔧 FIX 2: Handle infinite values - replace with NaN then drop
    X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.dropna()
    
    if X_clean.empty or len(X_clean) < 2:
        return pd.DataFrame({"Feature": X.columns, "VIF": [np.inf] * len(X.columns)})
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_scores = []
    
    for i in range(X_clean.shape[1]):
        try:
            # 🔧 FIX 3: Wrap in try-except to handle any VIF calculation errors
            vif = variance_inflation_factor(X_clean.values, i)
            vif_scores.append(float(vif))
        except (ValueError, ZeroDivisionError, Exception) as e:
            # 🔧 FIX 4: Skip problematic features gracefully
            print(f"Warning: VIF calculation failed for {X_clean.columns[i]} - {str(e)}")
            vif_scores.append(np.inf)
    
    vif_data["VIF"] = vif_scores
    return vif_data