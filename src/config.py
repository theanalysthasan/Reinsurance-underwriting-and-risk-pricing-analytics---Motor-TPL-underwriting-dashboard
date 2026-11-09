"""Central assumptions. Change here, re-run pipeline."""
SEED = 42
N_POLICIES = 120_000
LARGE_LOSS_THRESHOLD = 50_000      # claims above this are capped in the severity GLM and loaded separately
EXPENSE_RATIO = 0.25               # expenses as % of gross premium
PROFIT_MARGIN = 0.05               # target underwriting margin
RISK_LOAD_K = 1.0                 # total risk load = K * portfolio std dev, allocated to policies by variance share
# Excess-of-loss treaty (per claim): LAYER_LIMIT xs RETENTION
RETENTION = 100_000
LAYER_LIMIT = 400_000
REINS_EXPENSE_RATIO = 0.10
REINS_RISK_LOAD = 0.20             # load on std dev of layer loss
