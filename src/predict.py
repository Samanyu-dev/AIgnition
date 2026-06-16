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

with open(args.features, 'rb') as f:
    features = pickle.load(f)
with open(args.model, 'rb') as f:
    models = pickle.load(f)

today = str(date.today())
rows  = []
print("\n=== [PHASE 3] PROBABILISTIC FORECASTING PIPELINE ===")

# ── CHANNEL-LEVEL ROWS ──────────────────────────────────────
print("[PREDICT] Generating Channel-Level Forecasts...")
for ch in ['Google','Meta','Bing']:
    for horizon in [30, 60, 90]:
        p10, p50, p90 = bootstrap_forecast(models[ch], horizon)
        spend = models[ch]['recent_avg_spend'] * (horizon // 7)
        roas  = p50 / max(spend, 1)
        rows.append({
            'forecast_date':  today,
            'channel':        ch,
            'campaign_type':  'CHANNEL_TOTAL',
            'campaign_name':  'CHANNEL_TOTAL',
            'horizon_days':   horizon,
            'revenue_p10':    round(p10, 2),
            'revenue_p50':    round(p50, 2),
            'revenue_p90':    round(p90, 2),
            'roas_p50':       round(roas, 4),
            'estimated_spend':round(spend, 2),
        })

# ── CAMPAIGN-TYPE LEVEL ROWS ─────────────────────────────────
print("[PREDICT] Generating Campaign-Type Level Forecasts...")
camp = features['camp_baselines']
for ch in ['Google','Meta','Bing']:
    ch_camp = camp[camp['channel']==ch]
    camp_types = ch_camp.groupby('campaign_type').agg(
        total_rev   = ('total_rev','sum'),
        total_spend = ('total_spend','sum'),
        n_days      = ('n_days','sum'),
    ).reset_index()
    camp_types['daily_rev']   = camp_types['total_rev']   / camp_types['n_days'].clip(lower=1)
    camp_types['daily_spend'] = camp_types['total_spend'] / camp_types['n_days'].clip(lower=1)
    camp_types['roas']        = camp_types['total_rev']   / camp_types['total_spend'].clip(lower=0.01)
    
    for horizon in [30, 60, 90]:
        for _, r in camp_types.iterrows():
            p50   = r['daily_rev'] * horizon
            p10   = p50 * 0.55
            p90   = p50 * 1.55
            spend = r['daily_spend'] * horizon
            rows.append({
                'forecast_date':  today,
                'channel':        ch,
                'campaign_type':  r['campaign_type'],
                'campaign_name':  'TYPE_TOTAL',
                'horizon_days':   horizon,
                'revenue_p10':    round(p10, 2),
                'revenue_p50':    round(p50, 2),
                'revenue_p90':    round(p90, 2),
                'roas_p50':       round(r['roas'], 4),
                'estimated_spend':round(spend, 2),
            })

# ── CAMPAIGN-LEVEL ROWS ───────────────────────────────────────
print("[PREDICT] Generating Campaign-Level Forecasts...")
for _, r in camp.iterrows():
    for horizon in [30, 60, 90]:
        p50   = r['daily_rev'] * horizon
        p10   = p50 * 0.55
        p90   = p50 * 1.55
        spend = r['daily_spend'] * horizon
        rows.append({
            'forecast_date':  today,
            'channel':        r['channel'],
            'campaign_type':  r['campaign_type'],
            'campaign_name':  r['campaign_name'],
            'horizon_days':   horizon,
            'revenue_p10':    round(p10, 2),
            'revenue_p50':    round(p50, 2),
            'revenue_p90':    round(p90, 2),
            'roas_p50':       round(r['roas'], 4),
            'estimated_spend':round(spend, 2),
        })

# ── GRAND TOTAL ROWS ─────────────────────────────────────────
print("[PREDICT] Aggregating Grand Totals...")
df = pd.DataFrame(rows)
ch_totals = df[df['campaign_name']=='CHANNEL_TOTAL']
for horizon in [30, 60, 90]:
    ht = ch_totals[ch_totals['horizon_days']==horizon]
    rows.append({
        'forecast_date':  today,
        'channel':        'ALL',
        'campaign_type':  'GRAND_TOTAL',
        'campaign_name':  'GRAND_TOTAL',
        'horizon_days':   horizon,
        'revenue_p10':    round(ht['revenue_p10'].sum(), 2),
        'revenue_p50':    round(ht['revenue_p50'].sum(), 2),
        'revenue_p90':    round(ht['revenue_p90'].sum(), 2),
        'roas_p50':       round(ht['revenue_p50'].sum() / max(ht['estimated_spend'].sum(),1), 4),
        'estimated_spend':round(ht['estimated_spend'].sum(), 2),
    })

# ── WRITE OUTPUT ─────────────────────────────────────────────
print("[PREDICT] Compiling final submission file...")
df_out = pd.DataFrame(rows)
os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
df_out.to_csv(args.output, index=False)
print(f"[PREDICT] SUCCESS! Written {len(df_out)} rows to {args.output}")
