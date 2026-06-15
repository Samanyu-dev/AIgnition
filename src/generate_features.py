import argparse, glob, os, pickle, pandas as pd, numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./data')
parser.add_argument('--out', default='features.pkl')
args = parser.parse_args()

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
weekly_channel['roas'] = weekly_channel['revenue'] / weekly_channel['spend'].clip(lower=0.01)
weekly_channel['week_of_year'] = weekly_channel['week_start'].dt.isocalendar().week.astype(int)

channel_seasonality = {}
for ch in ['Google','Meta','Bing']:
    w = weekly_channel[weekly_channel['channel']==ch].copy()
    global_mean = w['revenue'].mean()
    if global_mean > 0:
        sea = w.groupby('week_of_year')['revenue'].mean() / global_mean
        channel_seasonality[ch] = sea.to_dict()
    else:
        channel_seasonality[ch] = {}

channel_baselines = {}
for ch in ['Google','Meta','Bing']:
    w = weekly_channel[weekly_channel['channel']==ch].sort_values('week_start')
    recent = w.tail(12)
    channel_baselines[ch] = {
        'recent_roas':         recent['revenue'].sum() / max(recent['spend'].sum(), 1),
        'recent_avg_spend':    recent['spend'].mean(),
        'recent_avg_revenue':  recent['revenue'].mean(),
        'n_weeks':             len(w),
    }

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
# Campaigns with zero revenue keep roas=0 — that is correct, not an error

features = {
    'all_df':              all_df,
    'weekly_channel':      weekly_channel,
    'channel_baselines':   channel_baselines,
    'channel_seasonality': channel_seasonality,
    'camp_baselines':      camp_baselines,
}
with open(args.out, 'wb') as f:
    pickle.dump(features, f)
print(f"Features saved to {args.out}")
print(f"  Rows: {len(all_df)}, Campaigns: {all_df['campaign_name'].nunique()}")
