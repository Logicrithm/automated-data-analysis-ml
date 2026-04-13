import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def detect_multicollinearity(X):
    '''Return a DataFrame of VIF for each feature '''
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data
