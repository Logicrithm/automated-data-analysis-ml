import pandas as pd
import numpy as np

class FeatureAnalysis:
    def __init__(self, data):
        self.data = data.fillna(0)

    def analyze(self):
        # Limit redundant pairs
        feature_pairs = []
        for i in range(len(self.data.columns)):
            for j in range(i + 1, len(self.data.columns)):
                if len(feature_pairs) < 10:
                    feature_pairs.append((self.data.columns[i], self.data.columns[j]))
                else:
                    break
        return feature_pairs

# Usage:
# fa = FeatureAnalysis(data)
# print(fa.analyze())