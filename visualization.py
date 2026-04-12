from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_plot(path: Path) -> str:
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def generate_visualizations(df: pd.DataFrame, target_column: str, output_dir: str) -> Dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    visuals: Dict[str, str] = {}

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        heatmap = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
        ax.set_yticks(range(len(corr.columns)))
        ax.set_yticklabels(corr.columns, fontsize=8)
        fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Correlation Heatmap")
        visuals["heatmap"] = _save_plot(output_path / "correlation_heatmap.png")

    if target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        plt.figure(figsize=(8, 4))
        plt.hist(df[target_column].dropna(), bins=30, color="#4C72B0", edgecolor="white")
        plt.title(f"{target_column} Distribution")
        plt.xlabel(target_column)
        plt.ylabel("Count")
        visuals["distribution"] = _save_plot(output_path / f"{target_column}_distribution.png")

    if "sqft_living" in df.columns and target_column in df.columns:
        scatter_df = df[["sqft_living", target_column]].dropna()
        if not scatter_df.empty:
            plt.figure(figsize=(8, 5))
            plt.scatter(scatter_df["sqft_living"], scatter_df[target_column], alpha=0.35, s=20, color="#DD8452")
            plt.title(f"sqft_living vs {target_column}")
            plt.xlabel("sqft_living")
            plt.ylabel(target_column)
            visuals["scatter_sqft_vs_price"] = _save_plot(output_path / "sqft_living_vs_price.png")

    return visuals
