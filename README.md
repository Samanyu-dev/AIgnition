# Probabilistic Revenue Forecasting for E-commerce Marketing — AIgnition 3.0

**Team Name:** [Your Team Name]
**Team Members:** [Name 1], [Name 2]
**College:** [Your College]

**Python Version:** 3.11

**How to run:**
```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
```

## Assumptions & Limitations
- Attribution data from each platform is taken as-is. No custom attribution engine is built.
- Channels are modeled independently. Cross-channel cannibalization is not modeled.
- Forecasts assume future budget pacing matches recent historical pacing unless a different budget is explicitly specified.
- Seasonality is derived from the 2024–2026 weekly pattern. It cannot predict future macro events, competitor actions, or new campaign launches.
- Bing forecasts have extremely high uncertainty due to the ROAS collapse in early 2026. Treat them with caution.
- The model is trained on data up to June 2026. Test data will be from the same schema.
