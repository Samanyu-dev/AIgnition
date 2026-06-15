# Run this ONCE offline: python src/generate_ai_insights.py
# Commit the resulting output/insights.json to the repo
# run.sh copies it to output/ at test time — does NOT call this script

import anthropic, json, os

client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

prompt = """You are a senior digital marketing analyst at NetElixir, a top e-commerce performance agency.

Analyze this multi-channel advertising data and provide structured insights.

PORTFOLIO SUMMARY:
- Google Ads: $9,266,677 total revenue | 4.76x overall ROAS | 6.17x recent ROAS | 92 campaigns
  - Top campaign: Search_TM_Campaign_01 = $1,270,238 lifetime at 6.32x ROAS
  - Channel types: Performance Max (63.4%), Search (25.1%), Shopping (11.4%)
  - Search outperforms PMax: 6.13x vs 4.58x ROAS

- Meta Ads: $1,656,750 total revenue | 8.44x overall ROAS | 6.59x recent ROAS | 16 campaigns
  - Best campaign: Remarketing_Brand_Campaign_03 = $380,949 at 18.33x ROAS
  - Prospecting_Brand_Campaign_02 = $399,551 at 13.22x ROAS
  - Recent ROAS trend: declining from ~13x to ~6.5x — audience saturation signal

- Bing Ads: $172,027 total revenue | 4.36x overall ROAS | 1.29x recent ROAS | 28 campaigns
  - CRITICAL: Revenue collapsed to near-zero in Feb-Mar 2026 while spend continued at $2K-$3K/month
  - ROAS dropped below 0.1x — structural failure, not seasonal variation
  - 22 of 28 campaigns have zero revenue across their entire history

90-DAY FORECAST:
- Google: P10=$365,185 | P50=$451,282 | P90=$578,359 | Blended ROAS=3.23x
- Meta: P10=$45,985 | P50=$87,679 | P90=$161,532 | Blended ROAS=5.34x
- Bing: P10=$6,567 | P50=$9,494 | P90=$12,734 | Blended ROAS=1.14x
- TOTAL: P10=$417,737 | P50=$548,455 | P90=$752,625 | Blended ROAS=3.34x

KEY ANOMALIES:
- Google Nov 25-Dec 1, 2024: $611,654 revenue (+5.1 sigma) — Black Friday 2024 peak
- Google Dec 1-14, 2025: $543K + $525K in consecutive weeks — consistent Q4 pattern
- Bing Nov-Dec 2024: ROAS spikes to 44-72x on tiny spend — likely attribution anomaly
- Bing Feb-Mar 2026: ROAS below 0.1x despite continued spend — structural breakdown

Return ONLY a JSON object with this exact structure (no markdown, no explanation outside JSON):
{
  "summary": "one sentence strategic verdict for the next 90 days",
  "drivers": [
    {"title": "...", "detail": "..."},
    {"title": "...", "detail": "..."},
    {"title": "...", "detail": "..."},
    {"title": "...", "detail": "..."}
  ],
  "risks": [
    {"title": "...", "detail": "..."},
    {"title": "...", "detail": "..."},
    {"title": "...", "detail": "..."},
    {"title": "...", "detail": "..."}
  ],
  "recommendations": [
    {"action": "...", "channel": "...", "impact": "..."},
    {"action": "...", "channel": "...", "impact": "..."},
    {"action": "...", "channel": "...", "impact": "..."}
  ]
}"""

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1500,
    messages=[{"role": "user", "content": prompt}]
)

raw = response.content[0].text
raw_clean = raw.replace('```json','').replace('```','').strip()
insights = json.loads(raw_clean)

os.makedirs('output', exist_ok=True)
with open('output/insights.json', 'w') as f:
    json.dump(insights, f, indent=2)
print("Saved to output/insights.json — commit this file to the repo")
