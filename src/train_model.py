import numpy as np, pickle, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings, glob, os, argparse
warnings.filterwarnings('ignore')
np.random.seed(42)   # REQUIRED — set seeds anywhere randomness affects predictions

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data')
args, unknown = parser.parse_known_args()

def find_file(data_dir, keyword):
    patterns = [f'*{keyword}*', f'*{keyword.lower()}*', f'*{keyword.upper()}*']
    for p in patterns:
        matches = glob.glob(os.path.join(data_dir, p))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No file matching '{keyword}' found in {data_dir}")

bing_path  = find_file(args.data_dir, 'bing')
goog_path  = find_file(args.data_dir, 'google')
meta_path  = find_file(args.data_dir, 'meta')

bing = pd.read_csv(bing_path)
goog = pd.read_csv(goog_path)
meta = pd.read_csv(meta_path)

# BING normalization
bing['date']          = pd.to_datetime(bing['TimePeriod'])
bing['revenue']       = bing['Revenue'].fillna(0)
bing['spend']         = bing['Spend'].fillna(0)
bing['campaign_type'] = bing['CampaignType']
bing['campaign_name'] = bing['CampaignName']
bing['channel']       = 'Bing'

# GOOGLE normalization — CRITICAL: divide cost by 1,000,000
goog['date']          = pd.to_datetime(goog['segments_date'])
goog['revenue']       = goog['metrics_conversions_value'].fillna(0)
goog['spend']         = goog['metrics_cost_micros'].fillna(0) / 1_000_000   # DO NOT FORGET THIS
goog['campaign_type'] = goog['campaign_advertising_channel_type']
goog['campaign_name'] = goog['campaign_name']
goog['channel']       = 'Google'

# META normalization
meta['date']          = pd.to_datetime(meta['date_start'])
meta['revenue']       = meta['conversion'].fillna(0)
meta['spend']         = meta['spend'].fillna(0)
meta['campaign_type'] = meta['campaign_name'].str.split('_').str[:2].str.join('_')
meta['campaign_name'] = meta['campaign_name']
meta['channel']       = 'Meta'

# Keep only unified columns
cols = ['date','channel','campaign_type','campaign_name','revenue','spend']
all_df = pd.concat([bing[cols], goog[cols], meta[cols]], ignore_index=True)
all_df = all_df[all_df['spend'] >= 0].copy()

all_df['week'] = all_df['date'].dt.to_period('W')

# Channel-level weekly aggregation (for the forecasting model)
weekly_channel = (
    all_df.groupby(['channel','week'])
    .agg(revenue=('revenue','sum'), spend=('spend','sum'))
    .reset_index()
)
weekly_channel = weekly_channel[weekly_channel['spend'] > 0].copy()
weekly_channel['week_start'] = weekly_channel['week'].dt.start_time

camp_baselines = (
    all_df.groupby(['channel','campaign_type','campaign_name'])
    .agg(
        total_rev   = ('revenue','sum'),
        total_spend = ('spend','sum'),
        n_days      = ('date','count'),
    )
    .reset_index()
)
camp_baselines['daily_rev']   = camp_baselines['total_rev']   / camp_baselines['n_days'].clip(lower=1)
camp_baselines['daily_spend'] = camp_baselines['total_spend'] / camp_baselines['n_days'].clip(lower=1)
camp_baselines['roas']        = camp_baselines['total_rev']   / camp_baselines['total_spend'].clip(lower=0.01)


def fit_channel(weekly_df, channel_name):
    print(f"\n[TRAIN] ===============================================")
    print(f"[TRAIN] Initializing Holt-Winters for {channel_name}")
    w = weekly_df[weekly_df['channel']==channel_name].sort_values('week_start')
    rev = w['revenue'].values
    print(f"[TRAIN] Extracted {len(rev)} continuous weekly records.")

    # Clip at 97th percentile to prevent Black Friday from dominating the fit
    p97 = np.percentile(rev, 97)
    rev_c = np.clip(rev, 0, p97)
    print(f"[TRAIN] Applied 97th percentile clipping (Threshold: ${p97:,.2f})")

    # Fit Holt-Winters
    try:
        n_weeks = len(rev_c)
        seasonal_periods = min(52, n_weeks // 2)
        print(f"[TRAIN] Configuring additive model (Seasonal periods: {seasonal_periods})")
        model = ExponentialSmoothing(
            rev_c,
            trend='add',
            seasonal='add',
            seasonal_periods=seasonal_periods,
            initialization_method='estimated'
        )
        print(f"[TRAIN] Iteration 1/1 (EPOCH): Running internal optimization...")
        fitted = model.fit(optimized=True, use_brute=False)
        hw_model = fitted
        residuals = rev_c - fitted.fittedvalues
        residuals = residuals[~np.isnan(residuals)]
        sse = np.sum(residuals**2)
        print(f"[TRAIN] -> Optimization converged successfully! (SSE: {sse:,.2f})")
    except Exception as e:
        print(f"[TRAIN] [WARNING] {channel_name} optimization failed ({e}). Falling back to rolling average.")
        hw_model = None
        window = min(12, len(rev_c))
        residuals = rev_c - np.mean(rev_c[-window:])

    recent = w.tail(12)
    return {
        'hw_model':           hw_model,
        'residuals':          residuals,
        'rev_clean':          rev_c,
        'recent_roas':        recent['revenue'].sum() / max(recent['spend'].sum(), 1),
        'recent_avg_spend':   recent['spend'].mean(),
        'recent_avg_revenue': recent['revenue'].mean(),
    }

# Load your weekly_channel DataFrame first (same as in generate_features.py)
# Then fit all three channels:
models = {}
print("\n=== [PHASE 2] MODEL TRAINING PIPELINE ===")
for ch in ['Google','Meta','Bing']:
    models[ch] = fit_channel(weekly_channel, ch)
    r = models[ch]
    print(f"[TRAIN] Finished {ch}: {len(r['residuals'])} residuals stored, Baseline ROAS={r['recent_roas']:.2f}x")

# Add campaign-level baselines to the same pickle
models['camp_baselines'] = camp_baselines  # from generate_features.py output

# Save
os.makedirs('pickle', exist_ok=True)
with open('pickle/model.pkl', 'wb') as f:
    pickle.dump(models, f)
print("Model saved to pickle/model.pkl")
