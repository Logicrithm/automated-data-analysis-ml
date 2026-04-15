import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def detect_multicollinearity(X):
    '''Return a DataFrame of VIF for each feature '''
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_scores = []
    for i in range(X.shape[1]):
        try:
            vif_scores.append(float(variance_inflation_factor(X.values, i)))
        except Exception:
            vif_scores.append(float("inf"))
    vif_data['VIF'] = vif_scores
    return vif_data
