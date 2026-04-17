# Automated Data Analysis ML Pipeline

Intelligent system for automated data analysis, pattern detection, and ML model interpretation.

## Features
✅ Data quality assessment  
✅ Context/domain inference (real estate / healthcare / business / generic)  
✅ Rule-chained recommendations with priority  
✅ Weighted confidence scoring (finding/reliability/domain/actionable/overall)  
✅ Linear + Random Forest + XGBoost comparison  
✅ Enhanced root-cause chain reasoning  
✅ Professional HTML reports  
✅ Model failure diagnosis

## Usage
```bash
python analyze.py <data.csv> [output_dir] [target_column]
```

## Results
- Identifies model performance issues
- Ranks insights by importance
- Provides actionable recommendations

## Experimental Validation Layer
Use the minimal experiment pipeline to validate recommendations with proof:

```python
import pandas as pd
from src.experiments import ExperimentPipeline

df = pd.read_csv("data.csv")
X = df.drop(columns=["target"])
y = df["target"]

pipeline = ExperimentPipeline(X, y)
pipeline.run_all()
results = pipeline.get_structured_results()
print(results["best_method"], results["improvement_percent"])
```

Included experiments:
- Baseline model score
- Remove high-VIF features (threshold 5.0)
- PCA (95% explained variance)
- Simple feature engineering (interaction + polynomial)
