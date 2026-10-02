# Notebooks

Exploratory analysis and prototyping live here. Anything that proves useful
should graduate into `src/` (e.g. a new indicator → `src/data/indicators.py`,
a new chart → `src/dashboard/components.py`).

Suggested starter: load `RELIANCE.NS` via `StockDataFetcher`, enrich with
`add_all`, and plot with the components from `src/dashboard/components.py`.
