"""Generate a synthetic motor third-party liability (casualty) portfolio.
Synthetic by design: structure mimics public MTPL datasets (e.g. freMTPL2) but ground truth is known,
so model recovery can be validated. To use real data, drop data/policies.csv + data/claims.csv in place."""
import numpy as np, pandas as pd
from config import *

rng = np.random.default_rng(SEED)
n = N_POLICIES
regions = ["North", "South", "East", "West", "Urban-Core", "Rural"]
reg_eff = dict(zip(regions, [0.0, -0.05, 0.05, 0.0, 0.35, -0.25]))

pol = pd.DataFrame({
    "policy_id": np.arange(1, n + 1),
    "driver_age": np.clip(rng.normal(44, 15, n), 18, 85).astype(int),
    "vehicle_age": rng.integers(0, 20, n),
    "vehicle_power": rng.integers(4, 15, n),
    "bonus_malus": np.clip(rng.gamma(4, 20, n), 50, 230).astype(int),
    "region": rng.choice(regions, n, p=[.18, .17, .15, .15, .2, .15]),
    "exposure": np.clip(rng.beta(5, 1.5, n), 0.05, 1.0).round(3),
})
age = pol.driver_age
log_f = (-2.55 + 0.9 * np.exp(-(age - 18) / 8)
         + 0.0008 * (age - 44).clip(lower=0)
         + 0.012 * (pol.bonus_malus - 100) / 10
         + 0.03 * (pol.vehicle_power - 8)
         - 0.015 * pol.vehicle_age
         + pol.region.map(reg_eff))
true_freq = np.exp(log_f)
pol["n_claims"] = rng.poisson(true_freq * pol.exposure)

rows = []
for pid, k, a, p, reg in zip(pol.policy_id, pol.n_claims, pol.driver_age, pol.vehicle_power, pol.region):
    for _ in range(k):
        mean = 2_800 * (1 + 0.012 * (p - 8)) * (1.25 if reg == "Urban-Core" else 1.0) * (1.15 if a < 25 else 1.0)
        if rng.random() < 0.035:                         # large bodily-injury claim, Pareto tail
            amt = 30_000 * (1 + rng.pareto(1.6)) * (1.2 if reg == "Urban-Core" else 1.0)
        else:
            amt = rng.gamma(2.0, mean / 2.0)
        rows.append((pid, round(min(amt, 5_000_000), 2)))
claims = pd.DataFrame(rows, columns=["policy_id", "claim_amount"])
claims.insert(0, "claim_id", np.arange(1, len(claims) + 1))

# Current tariff: under-differentiated -> creates unprofitable pockets for the analysis to find
pol["gross_premium"] = ((640 + 0.9 * (pol.bonus_malus - 100) + 6 * (pol.vehicle_power - 8)
                         + np.where(age < 25, 120, 0)) * pol.exposure).round(2)
pol.to_csv("../data/policies.csv", index=False)
claims.to_csv("../data/claims.csv", index=False)
print(f"policies={len(pol):,} claims={len(claims):,} gross_premium={pol.gross_premium.sum():,.0f} "
      f"incurred={claims.claim_amount.sum():,.0f} large(>{LARGE_LOSS_THRESHOLD})={(claims.claim_amount>LARGE_LOSS_THRESHOLD).sum()}")
