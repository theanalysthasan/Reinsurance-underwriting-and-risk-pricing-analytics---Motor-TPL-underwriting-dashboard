# P&C Reinsurance Underwriting & Risk Pricing Analytics
Python · SQL · Power BI · Poisson/Gamma GLM · XGBoost · Monte Carlo

End-to-end casualty (motor TPL) pricing and treaty analysis: frequency × severity → pure premium → risk-loaded technical premium → excess-of-loss reinsurance → segment profitability → dashboard.

> **Data note:** the portfolio is **synthetic** (120k policies, ~7.8k claims, Pareto-tailed bodily-injury losses) with known ground truth and a deliberately under-differentiated tariff. It is not client data. Swap in real data (e.g. freMTPL2) by replacing `data/policies.csv` and `data/claims.csv`.

## Run
```
pip install -r requirements.txt
./run_all.sh          # generates data, fits models, runs reinsurance + profitability, writes /outputs
```

## Method
| Step | Script | Approach |
|---|---|---|
| Frequency | `02_pricing_models.py` | Poisson GLM, log-exposure offset; XGBoost Poisson benchmark; holdout Poisson deviance |
| Severity | same | Gamma GLM (log link) on losses capped at 50k + separate large-loss load |
| Pure premium | same | frequency × severity |
| Technical premium | same | (pure premium + risk load) / (1 − expense − margin). Risk load = std-dev principle at portfolio level, allocated by variance share (compound-Poisson variance) |
| XoL treaty | `03_reinsurance_portfolio.py` | 400k xs 100k per claim: experience burning cost + Monte Carlo (empirical body, Pareto tail MLE); retention comparison 50k–250k |
| Profitability | same | Loss/combined ratio, rate adequacy, indicated rate change by region × age band; action flags |
| SQL | `sql/segment_analysis.sql` | Segment loss ratios, layer experience, frequency/severity, under-priced policies |

## Results (synthetic data, seed 42)
- Poisson GLM holdout deviance 0.3455 vs XGBoost 0.3460 (null 0.3548): the GLM matches the ML benchmark and stays interpretable.
- Portfolio loss ratio 70%; 13 of 30 segments have combined ratio > 100%. Worst: Urban-Core 18-24 (CR ≈ 204%, rate adequacy 0.39). Rural 35+ is over-priced (adequacy 1.5-1.7).
- 400k xs 100k: 36 claims pierce retention; burning cost 5.5% of GWP; simulated expected layer loss ≈ 2.9m, technical reinsurance premium ≈ 3.4m.
- Lowering retention to 50k cuts annual-loss volatility ~30% vs ~15% at 250k, at higher net cost of cover.
- Caveat: small segments (e.g. 65+) are volatile because of single large claims; production use needs credibility weighting.

