import argparse, pickle, pandas as pd, numpy as np, os, warnings
from datetime import date
warnings.filterwarnings('ignore')
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--features', default='features.pkl')
parser.add_argument('--model',    default='./pickle/model.pkl')
parser.add_argument('--output',   default='./output/predictions.csv')
args, unknown = parser.parse_known_args()

def budget_scale_factor(base_spend_total, new_budget_total):
    """
    Log diminishing returns curve.
    Calibrated on e-commerce industry norms:
    - Near linear below 1.5x baseline
    - Meaningful diminishing returns above 1.5x
    - Underspend hurts less proportionally than overspend helps
    """
    if base_spend_total <= 0:
        return 1.0
    ratio = new_budget_total / base_spend_total
    if ratio <= 0:
        return 0.0
    elif ratio < 0.5:
        return ratio ** 0.6    # underspend: revenue drops less than spend ratio
    elif ratio <= 1.5:
        return ratio ** 0.85   # near base: roughly linear, slight diminishing returns
    else:
        return ratio ** 0.55   # significant overspend: strong diminishing returns

def bootstrap_forecast(model_entry, horizon_days, new_budget=None, n_boot=2000):
    """
    Generates P10/P50/P90 revenue forecast for one channel and one horizon.
    """
    hw      = model_entry['hw_model']
    resid   = model_entry['residuals']
    rev_c   = model_entry['rev_clean']
    horizon_weeks = max(1, horizon_days // 7)

    if hw is not None:
        base = np.maximum(hw.forecast(horizon_weeks), 0)
    else:
        avg = np.mean(rev_c[-12:]) if len(rev_c) >= 12 else np.mean(rev_c)
        base = np.full(horizon_weeks, max(avg, 0))

    if new_budget is not None:
        base_spend = model_entry['recent_avg_spend'] * horizon_weeks
        scale = budget_scale_factor(base_spend, new_budget)
        base = base * scale

    boot_totals = []
    for _ in range(n_boot):
        sampled_resid = np.random.choice(resid, size=horizon_weeks, replace=True)
        scenario_total = np.maximum(base + sampled_resid, 0).sum()
        boot_totals.append(scenario_total)

    boot_totals = np.array(boot_totals)
    p10 = max(np.percentile(boot_totals, 10), 0)
    p50 = max(np.percentile(boot_totals, 50), 0)
    p90 = max(np.percentile(boot_totals, 90), 0)
    return p10, p50, p90
