# 🚀 AI-Powered Indian Stock Market Dashboard

> A premium, AI-powered analytics terminal for Indian equities (NSE/BSE) — live price data, technical indicators, and AI-generated **Top-5 trade ideas** in a Zerodha/Upstox-style dark UI. Runs **100% free** on Streamlit Community Cloud.

---

## ✨ Features

- 🏆 **Top 5 AI Picks** — scans the top-30 NIFTY universe, ranks the 15 biggest movers by absolute daily % change, and asks an LLM for 5 trade ideas (entry, 3 targets, stop-loss) returned in **strict JSON**
- 🌑 **Premium Dark UI** — trading-terminal look: gradient hero header, KPI cards, dark candlestick & RSI/MACD charts, hoverable recommendation cards
- 📊 **Technical Indicators** — SMA(20/50), EMA, RSI(14), MACD, Bollinger Bands computed with pandas
- 🤖 **AI Stock Analysis** — per-stock natural-language analysis from any OpenAI-compatible LLM (OpenAI, ZCode/GLM gateway, local servers)
- 🛟 **Graceful Fallback Mode** — no API key? The app still works: a deterministic rule-based engine produces recommendations and summaries locally
- 🖥️ **CLI + Offline Tests** — `python main.py fetch / analyze / recommend` and a 28-test suite that needs no network
- 🌐 **Free Deployment** — one public GitHub repo + [Streamlit Community Cloud](https://share.streamlit.io), no credit card required

---

## 📋 Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.9+** | 3.10–3.12 recommended (project is developed/tested on 3.12) |
| **Git** | for cloning and cloud deployment |
| **API key** *(optional)* | any OpenAI-compatible key — only needed for AI features; the app falls back to local heuristics without one |

---

## ⚙️ Local Setup (step by step)

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Configure API keys (optional)

Two supported options — **either** works locally. Both files are gitignored, so real keys are never committed.

**Option A — `.env` file (classic)**

```bash
cp .env.example .env
```

Then edit `.env`:

```env
OPENAI_API_KEY=sk-your-real-key
OPENAI_BASE_URL=https://your-gateway.example.com/v1   # only for gateways (ZCode/GLM etc.)
OPENAI_MODEL=gpt-4o-mini                              # optional
```

**Option B — `.streamlit/secrets.toml` (same format as the cloud)**

A commented template already exists at `.streamlit/secrets.toml` — just edit it:

```toml
OPENAI_API_KEY = "your-real-key"
OPENAI_BASE_URL = "https://your-gateway.example.com/v1"
```

> 💡 **No key at all?** Skip this step — the dashboard runs in **fallback mode** (heuristic recommendations, rule-based summaries). Placeholder values like `"your_..."` are detected and treated as "not configured", so you'll never hit confusing auth errors.

### 5️⃣ Run the app

```bash
streamlit run src/dashboard/app.py
```

Open the shown URL (default `http://localhost:8501`). Port busy? Add `--server.port 8502`.

---

## 🧭 Using the Dashboard

| Action | How |
|---|---|
| Analyze a stock | Sidebar → pick symbol & period → **Load data** → charts + KPI strip update |
| Per-stock AI view | **Generate AI insights** button in the *AI Analysis* section |
| Market-wide Top-5 | Sidebar → **Get Top 5 Picks** (first scan takes ~1–2 min: 30 symbols are fetched) |
| Read a pick card | Badge = trade horizon · green = entry · blue boxes = targets T1–T3 · red = stop-loss · R:R = reward-to-risk |

**CLI alternatives:**

```bash
python main.py fetch RELIANCE     # download & preview price history
python main.py analyze TCS        # indicator snapshot + AI insights
python main.py recommend          # Top-5 picks as JSON
```

**Run the tests** (offline, no network needed):

```bash
pytest          # or: python -m pytest
```

---

## 🌐 FREE Cloud Deployment — Streamlit Community Cloud

Deploy the exact same dashboard in ~5 minutes, free forever.

### Step 1 — Push the code to a public GitHub repository

```bash
# inside the project folder
git init
git add .
git commit -m "AI Stock Dashboard: initial release"

# create a NEW PUBLIC repo on https://github.com/new (no README/.gitignore — we have them)
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

> ⚠️ The repo must be **public** for the free tier. Double-check that `.env` and `.streamlit/secrets.toml` were **not** pushed (they're gitignored — `git status` should not list them).

### Step 2 — Sign in to Streamlit Community Cloud

Go to **[share.streamlit.io](https://share.streamlit.io)** and click **Sign in with GitHub** (authorize Streamlit when prompted).

### Step 3 — Create the app

1. Click **Create app** (or **New app**) → **Paste GitHub repo URL** (or pick the repo from the dropdown).
2. **Repository:** `<your-username>/<repo-name>`
3. **Branch:** `main`
4. **Main file path:** `src/dashboard/app.py` ← *exactly this*
5. *(Optional)* Advanced settings → Python version **3.12**
6. Click **Deploy!**

Dependencies install automatically from `requirements.txt` (first build ≈ 3–5 min).

### Step 4 — ⚠️ CRITICAL: add your API secrets

Without this step the app runs, but stays in fallback mode (no real AI).

1. In your app, click the **⋮** menu (bottom-right) → **Settings** → **Secrets**.
2. Paste the following into the secrets editor, **replacing the values with your real credentials**:

```toml
OPENAI_API_KEY = "your_zcode_or_glm_api_key_here"
OPENAI_BASE_URL = "your_api_base_url_here"
```

3. Click **Save** — the app reboots automatically and the AI features go live.

> 📌 Notes:
> - `OPENAI_BASE_URL` is required for OpenAI-compatible gateways (ZCode/GLM etc.) and must include `/v1` at the end for most providers. **Plain OpenAI?** You can omit `OPENAI_BASE_URL` entirely.
> - Optional extra line: `OPENAI_MODEL = "gpt-4o-mini"` to pin the model.
> - Secrets are encrypted at rest and never appear in your repo.

### Step 5 — What to expect

- Every `git push` to `main` **redeploys the app automatically**.
- Free-tier apps **sleep after inactivity** and wake on the next visit (first wake takes ~a minute).
- Free resources: 1 vCPU / ~1 GB RAM — plenty for this app; the market scan may take a bit longer than locally.
- Debug anytime via **Manage app → Logs**.

---

## 🛠️ Troubleshooting

| Symptom | Cause & Fix |
|---|---|
| `ModuleNotFoundError: No module named 'config'` | Wrong **Main file path** on Streamlit Cloud. It must be exactly `src/dashboard/app.py` — the app bootstraps its own import path from there. |
| Insights/picks say *"heuristic"* or *"configure OPENAI_API_KEY"* | No key, or a placeholder key was left in place. Set real values in **Settings → Secrets** (cloud) or `.env` (local). Placeholder strings like `"your_..."` are treated as unconfigured **by design**. |
| `ValueError: If using all scalar values, you must pass an index` | Older bug with yfinance's MultiIndex columns — already fixed in `src/data/fetcher.py`. If you still see it: `pip install -U yfinance` and redeploy. |
| Yahoo errors: `401 Invalid Crumb`, `No data found, symbol may be delisted` | Transient Yahoo-side issues or delisted tickers (e.g. TATAMOTORS was removed from the universe for this reason). The scanner **skips** failing symbols and continues as long as ≥5 succeed — retry later if it persists. |
| `LLM call failed: 401 / model not found` | Wrong `OPENAI_BASE_URL` (most gateways need `/v1` at the end), wrong model name, or invalid key. The app falls back to heuristics instead of crashing. |
| Market scan returns fewer than 5 picks | Fewer than 5 symbols had valid data; the engine ranks whatever it fetched. Check the logs for skipped symbols. |
| Charts/blank screen right after deploy | First data fetch happens on load — give it ~30 s, then check **Manage app → Logs**. |
| Local: `Port 8501 is already in use` | `streamlit run src/dashboard/app.py --server.port 8502` |

---

## 📁 Project Structure

```
.
├── main.py                 # CLI entry point (fetch / analyze / recommend)
├── conftest.py             # makes bare `pytest` work from the repo root
├── config/
│   └── settings.py         # env + Streamlit-secrets settings, NIFTY universe
├── src/
│   ├── data/
│   │   ├── fetcher.py      # yfinance wrapper (NSE/BSE symbol mapping, column flattening)
│   │   └── indicators.py   # SMA, EMA, RSI, MACD, Bollinger Bands
│   ├── ai/
│   │   ├── llm_client.py   # OpenAI-compatible client (base-URL aware, secrets-aware)
│   │   ├── analyzer.py     # per-stock AI insights (+ heuristic fallback)
│   │   └── recommender.py  # Top-5 picks: movers scan → strict-JSON AI output
│   ├── dashboard/
│   │   ├── app.py          # Streamlit app — MAIN FILE for cloud deployment
│   │   └── components.py   # dark-themed charts, KPI cards, recommendation cards
│   └── utils/
│       └── logger.py       # logging setup
├── .streamlit/
│   ├── config.toml         # dark theme config
│   └── secrets.toml        # template — gitignored, fill locally / paste on Cloud
├── tests/                  # 28 offline tests
├── data/                   # downloaded data (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## ⚠️ Disclaimer

> **This dashboard is provided strictly for educational and research purposes.**
>
> - It is **not** investment advice, and nothing generated by it (AI insights, Top-5 picks, targets, stop-losses) should be treated as a recommendation to buy or sell any security.
> - The authors and contributors are **not registered with the Securities and Exchange Board of India (SEBI)** as investment advisers or research analysts.
> - Market data is sourced from third parties (Yahoo Finance), may be delayed, incomplete or inaccurate.
> - Trading in equities involves substantial risk of loss. Always consult a **SEBI-registered investment adviser** before making investment decisions.
