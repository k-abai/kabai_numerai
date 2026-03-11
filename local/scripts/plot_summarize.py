import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from numerapi import NumerAPI
from numerai_tools.scoring import numerai_corr, correlation_contribution, neutralize

def compute_perf_metrics(series: pd.Series) -> dict:
    series = pd.Series(np.asarray(series).reshape(-1)).dropna()

    mean = series.mean()
    std = series.std(ddof=0)
    sharpe = mean / std if std != 0 else np.nan

    cum = series.cumsum()
    running_max = cum.expanding(min_periods=1).max()
    drawdown = running_max - cum
    max_drawdown = drawdown.max()

    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }

def plot_and_summarize_validation(
    per_era_corr: pd.Series,
    per_era_mmc: pd.Series,
    report_dir: str = "local/reports",
) -> pd.DataFrame:
    os.makedirs(report_dir, exist_ok=True)

    # Metrics
    corr_metrics = compute_perf_metrics(per_era_corr)
    mmc_metrics = compute_perf_metrics(per_era_mmc)

    metrics = pd.DataFrame({
        "CORR": corr_metrics,
        "MMC": mmc_metrics,
    })

    print("\nValidation Metrics")
    print(metrics.round(6))

    metrics_path = os.path.join(report_dir, "validation_metrics.csv")
    metrics.to_csv(metrics_path)
    print(f"\nSaved metrics to: {metrics_path}")

    # CORR plot
    per_era_corr.plot(
        title="Validation CORR",
        kind="bar",
        figsize=(8, 4),
        xticks=[],
        legend=False,
        snap=False,
    )
    plt.tight_layout()
    corr_plot_path = os.path.join(report_dir, "validation_corr.png")
    plt.savefig(corr_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # MMC plot
    per_era_mmc.plot(
        title="Validation MMC",
        kind="bar",
        figsize=(8, 4),
        xticks=[],
        legend=False,
        snap=False,
    )
    plt.tight_layout()
    mmc_plot_path = os.path.join(report_dir, "validation_mmc.png")
    plt.savefig(mmc_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved CORR plot to: {corr_plot_path}")
    print(f"Saved MMC plot to: {mmc_plot_path}")

    return metrics