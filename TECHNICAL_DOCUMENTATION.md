# AIgnition 3.0: Technical Documentation
**Probabilistic Revenue Forecasting for E-commerce Marketing**

## Section 1: Forecasting Methodology

Our core forecasting engine relies on **Holt-Winters Additive Exponential Smoothing**. We specifically chose an additive model (where trend and seasonality are added rather than multiplied) because the seasonal variation in the provided advertising data remains roughly constant in absolute dollars rather than scaling proportionally to the baseline revenue level.

- **Seasonality:** We model a 52-week seasonal period to accurately capture annual recurring events, most notably the massive Q4 retail spikes.
- **Outlier Mitigation:** Before fitting the model, we clip the historical revenue at the 97th percentile. This prevents extreme, singular anomalies (like the +5.1 sigma Black Friday 2024 peak) from entirely dominating the baseline seasonal component.
- **Probabilistic Bootstrapping:** To generate realistic confidence intervals (P10, P50, P90), we do not rely on arbitrary percentage bands. Instead, we sample from the empirical distribution of the model's historical residuals 2,000 times per horizon. We sum these scenarios across the forecasted period and extract the 10th, 50th, and 90th percentiles to produce honest ranges grounded in actual data variance.

## Section 2: Data Preprocessing

Our preprocessing pipeline (`generate_features.py`) guarantees a clean, unified schema across all three heterogeneous data sources:

1. **Google Cost Normalization:** `metrics_cost_micros` is aggressively divided by 1,000,000 on load. Failure to do this inflates Google spend artificially, corrupting all ROAS calculations.
2. **Unified Schema:** We map platform-specific date, spend, and revenue columns into a strict, unified standard `['date', 'channel', 'campaign_type', 'campaign_name', 'revenue', 'spend']`.
3. **Null Handling:** Missing values (e.g., the 7 nulls in Meta's `daily_budget`) are robustly handled via `.fillna(0)` to prevent pipeline crashes on unseen test data.
4. **Zero-Spend Filtering:** Weeks with zero total spend are filtered out prior to training to prevent division-by-zero ROAS errors and skewed baselines.
5. **Weekly Aggregation:** We aggregate the data to a weekly frequency. This smooths out irrelevant daily noise while perfectly preserving the critical macro-seasonality required for 30/60/90-day aggregate forecasts.

## Section 3: Assumptions

1. Attribution data from each platform is taken as-is. No custom attribution engine is built.
2. Channels are modeled independently. Cross-channel cannibalization is not modeled.
3. Forecasts assume future budget pacing matches recent historical pacing unless a different budget is explicitly specified.
4. Seasonality is derived from the 2024–2026 weekly pattern. It cannot predict future macro events, competitor actions, or new campaign launches.
5. Bing forecasts have extremely high uncertainty due to the ROAS collapse in early 2026. Treat them with caution.
6. The model is trained on data up to June 2026. Test data will be from the same schema.

## Section 4: Limitations

1. **Bing Reliability:** Bing forecasts carry extremely high uncertainty due to a structural ROAS collapse in early 2026. The model reports what historical data implies, but the channel may continue to deteriorate.
2. **Missing Offline Data:** No offline conversion data from Shopify or internal CRM systems is incorporated.
3. **Q4 Volatility:** Q4 forecasts have extreme uncertainty. The P10-P90 range is very wide because Black Friday performance varies significantly year-to-year.
4. **Campaign-Level Bounds:** Campaign-level P10 and P90 ranges are derived from a fixed 55%/155% scaling of the P50 forecast. A more computationally expensive but sophisticated approach would use campaign-level residual distributions.
5. **Budget Adjustments:** The budget scenario adjustment uses a calibrated logarithmic diminishing returns curve based on industry norms, rather than an explicitly measured price elasticity curve for this specific client.

## Section 5: AI Integration Strategy

Our architecture strictly separates statistical forecasting from causal interpretation. The Holt-Winters model generates the quantitative forecasts, but we use an **AI Insights Layer (Claude / Anthropic API)** to provide the *qualitative "why"* behind the numbers. 

The LLM is provided with deep portfolio context, recent ROAS trends, anomaly history, and forecast uncertainty. It interprets these metrics to flag specific risks (e.g., audience saturation on Meta) and identify drivers (e.g., PMax vs. Search efficiency). 

**Offline Execution:** To guarantee stability and comply with testing constraints (no internet access during the automated scoring pipeline), the AI insights are pre-generated offline. The output is committed to the repository as `insights.json`. During test time, the pipeline simply surfaces this pre-computed strategic analysis alongside the live statistical predictions.
