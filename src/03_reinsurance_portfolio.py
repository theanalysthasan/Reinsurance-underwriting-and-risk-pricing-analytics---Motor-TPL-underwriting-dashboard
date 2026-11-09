"""Excess-of-loss reinsurance analysis (experience + Monte Carlo) and portfolio profitability by segment."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *

rng = np.random.default_rng(SEED)
pol = pd.read_csv("../outputs/policy_pricing.csv"); clm = pd.read_csv("../data/claims.csv")
layer = lambda x, r, l: np.clip(x - r, 0, l)

# ---------- A. Experience (burning cost) ----------
clm["ceded"] = layer(clm.claim_amount, RETENTION, LAYER_LIMIT)
clm["retained"] = clm.claim_amount - clm.ceded
prem = pol.gross_premium.sum()
print(f"[Experience] {LAYER_LIMIT:,} xs {RETENTION:,}: claims in layer={(clm.claim_amount>RETENTION).sum()}  "
      f"ceded={clm.ceded.sum():,.0f}  burning cost={clm.ceded.sum()/prem:.2%} of GWP")

# ---------- B. Monte Carlo: severity = empirical body + fitted Pareto tail ----------
thr = LARGE_LOSS_THRESHOLD
tail = clm.claim_amount[clm.claim_amount > thr].values
alpha = len(tail) / np.log(tail / thr).sum()                    # Hill / Pareto MLE
p_tail = len(tail) / len(clm)
body = clm.claim_amount[clm.claim_amount <= thr].values
exp_claims = pol.freq.mul(pol.exposure).sum()

def simulate(n_sims=5000, retention=RETENTION, limit=LAYER_LIMIT):
    out = np.empty((n_sims, 2))
    for i in range(n_sims):
        k = rng.poisson(exp_claims)
        is_tail = rng.random(k) < p_tail
        x = np.where(is_tail, thr * (1 - rng.random(k)) ** (-1 / alpha), rng.choice(body, k))
        x = np.minimum(x, 5_000_000)
        out[i] = (x.sum(), layer(x, retention, limit).sum())
    return out

sim = simulate()
agg, ced = sim[:, 0], sim[:, 1]
el, sd = ced.mean(), ced.std()
reins_prem = (el + REINS_RISK_LOAD * sd) / (1 - REINS_EXPENSE_RATIO)
print(f"[Simulation] Pareto alpha={alpha:.2f}  E[layer loss]={el:,.0f}  SD={sd:,.0f}  "
      f"P(layer loss>0)={(ced>0).mean():.1%}  VaR99.5 layer={np.percentile(ced,99.5):,.0f}  technical reinsurance premium={reins_prem:,.0f}")
pd.DataFrame({"metric": ["Pareto alpha", "Expected layer loss", "Layer loss std dev", "P(layer hit)", "VaR 99.5% layer loss",
                         "Technical reinsurance premium", "Rate on line", "Burning cost (experience)"],
              "value": [alpha, el, sd, (ced > 0).mean(), np.percentile(ced, 99.5), reins_prem, reins_prem / LAYER_LIMIT,
                        clm.ceded.sum()]}).to_csv("../outputs/reinsurance_summary.csv", index=False)

# Retention comparison: what the cedent gives up vs what volatility it removes
rows = []
for r in [50_000, 100_000, 150_000, 250_000]:
    s = simulate(2000, r, LAYER_LIMIT)
    net = s[:, 0] - s[:, 1]; e, sdv = s[:, 1].mean(), s[:, 1].std()
    rp = (e + REINS_RISK_LOAD * sdv) / (1 - REINS_EXPENSE_RATIO)
    rows.append(dict(retention=r, limit=LAYER_LIMIT, expected_recovery=e, reinsurance_premium=rp,
                     net_cost_of_cover=rp - e, gross_std=s[:, 0].std(), net_std=net.std(),
                     vol_reduction=1 - net.std() / s[:, 0].std(), loss_ratio_on_layer=e / rp))
ret = pd.DataFrame(rows); ret.to_csv("../outputs/retention_options.csv", index=False)
print(ret.round(3).to_string(index=False))

# ---------- C. Portfolio profitability ----------
pol["segment"] = pol.region + " | " + pol.age_band.astype(str)
g = pol.groupby("segment").agg(policies=("policy_id", "count"), exposure=("exposure", "sum"),
        premium=("gross_premium", "sum"), incurred=("incurred", "sum"), expected_loss=("pure_premium", "sum"),
        technical_premium=("technical_premium", "sum")).reset_index()
g["loss_ratio"] = g.incurred / g.premium
g["expected_loss_ratio"] = g.expected_loss / g.premium
g["combined_ratio"] = g.loss_ratio + EXPENSE_RATIO
g["rate_adequacy"] = g.premium / g.technical_premium
g["rate_change_needed"] = 1 / g.rate_adequacy - 1
g["uw_result"] = g.premium * (1 - EXPENSE_RATIO) - g.incurred
def act(r):
    if r.rate_adequacy < 0.85: return "Re-price / restrict"
    if r.rate_adequacy < 0.97: return "Rate increase"
    if r.rate_adequacy > 1.15: return "Grow / defend"
    return "Maintain"
g["recommendation"] = g.apply(act, axis=1)
g.sort_values("rate_adequacy").to_csv("../outputs/segment_profitability.csv", index=False)
tot = g[["premium", "incurred", "expected_loss", "technical_premium"]].sum()
print(f"[Portfolio] LR={tot.incurred/tot.premium:.1%} CR={tot.incurred/tot.premium+EXPENSE_RATIO:.1%}  "
      f"unprofitable segments (CR>100%)={(g.combined_ratio>1).sum()}/{len(g)}")
print(g.sort_values("rate_adequacy")[["segment","premium","loss_ratio","rate_adequacy","recommendation"]].head(6).round(3).to_string(index=False))
print(g.sort_values("rate_adequacy")[["segment","premium","loss_ratio","rate_adequacy","recommendation"]].tail(4).round(3).to_string(index=False))

# ---------- Charts ----------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
h = g.pivot_table(index=g.segment.str.split(" \\| ").str[0], columns=g.segment.str.split(" \\| ").str[1],
                  values="combined_ratio")
im = ax[0].imshow(h.values, cmap="RdYlGn_r", vmin=.6, vmax=1.5); ax[0].set_xticks(range(h.shape[1]), h.columns)
ax[0].set_yticks(range(h.shape[0]), h.index); ax[0].set_title("Combined ratio by segment"); plt.colorbar(im, ax=ax[0])
for (i, j), v in np.ndenumerate(h.values): ax[0].text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8)
ax[1].hist(ced, bins=60, color="#2a6f97"); ax[1].set_yscale("log")
ax[1].set_title(f"Simulated annual XoL layer loss\n{LAYER_LIMIT/1e3:.0f}k xs {RETENTION/1e3:.0f}k"); ax[1].set_xlabel("Layer loss")
ax[2].bar(ret.retention.astype(str), ret.vol_reduction * 100, color="#6a994e")
ax[2].set_title("Volatility reduction vs retention"); ax[2].set_ylabel("% reduction in std dev of annual loss")
plt.tight_layout(); plt.savefig("../outputs/dashboard_preview.png", dpi=130)
