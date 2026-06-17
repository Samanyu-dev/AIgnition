import pandas as pd, numpy as np, pickle, os
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

print("=== [PHASE 1] BACKTESTING ENGINE ===")
if not os.path.exists('features.pkl'):
    print("Run ./run.sh first to generate features.")
    exit(1)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

df = features['weekly_channel']

print("Evaluating Holt-Winters Model on 8-Week Holdout...\n")

for ch in ['Google', 'Meta', 'Bing']:
    w = df[df['channel']==ch].sort_values('week_start')
    if len(w) < 16:
        continue
        
    train = w.iloc[:-8]
    test = w.iloc[-8:]
    
    rev = train['revenue'].values
    p97 = np.percentile(rev, 97)
    rev_c = np.clip(rev, 0, p97)
    
    seasonal_periods = min(52, len(rev_c) // 2)
    try:
        model = ExponentialSmoothing(
            rev_c, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=seasonal_periods, 
            initialization_method='estimated'
        )
        fitted = model.fit(optimized=True, use_brute=False)
        preds = fitted.forecast(8)
        preds = np.maximum(preds, 0)
        
        actuals = test['revenue'].values
        # Calculate MAPE, avoiding division by zero
        mape = np.mean(np.abs((actuals - preds) / np.maximum(actuals, 1))) * 100
        
        print(f"[{ch}] 8-Week Holdout MAPE: {mape:.2f}%")
        print(f"  Actuals:    {[int(x) for x in actuals]}")
        print(f"  Forecast:   {[int(x) for x in preds]}\n")
        
    except Exception as e:
        print(f"[{ch}] Backtest failed: {e}\n")
