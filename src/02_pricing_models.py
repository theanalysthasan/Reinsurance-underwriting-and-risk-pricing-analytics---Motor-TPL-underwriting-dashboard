"""Frequency (Poisson GLM + XGBoost), severity (Gamma GLM), pure premium, risk-based technical premium."""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance
import xgboost as xgb
from config import *

pol = pd.read_csv("../data/policies.csv"); clm = pd.read_csv("../data/claims.csv")
pol["age_band"] = pd.cut(pol.driver_age, [17, 24, 34, 49, 64, 120], labels=["18-24", "25-34", "35-49", "50-64", "65+"])
pol["young"] = (pol.driver_age < 25).astype(int)
pol["log_age"] = np.log(pol.driver_age - 17)
train, test = train_test_split(pol, test_size=0.3, random_state=SEED)
F = "n_claims ~ log_age + young + vehicle_age + vehicle_power + bonus_malus + C(region)"

# ---- 1. Frequency: Poisson GLM, log-exposure offset
freq = smf.glm(F, train, family=sm.families.Poisson(), offset=np.log(train.exposure)).fit()
test = test.copy()
test["freq_glm"] = freq.predict(test, offset=np.log(test.exposure)) / test.exposure   # annual rate

# ---- 1b. Frequency benchmark: XGBoost Poisson
feats = ["driver_age", "vehicle_age", "vehicle_power", "bonus_malus"]
X = lambda d: pd.concat([d[feats], pd.get_dummies(d.region, dtype=int)], axis=1)
dtr = xgb.DMatrix(X(train), label=train.n_claims, base_margin=np.log(train.exposure))
dte = xgb.DMatrix(X(test), label=test.n_claims, base_margin=np.log(test.exposure))
bst = xgb.train({"objective": "count:poisson", "max_depth": 3, "eta": 0.05, "subsample": .8, "seed": SEED},
                dtr, 300)
test["freq_xgb"] = bst.predict(dte) / test.exposure
dev = lambda col: mean_poisson_deviance(test.n_claims, test[col] * test.exposure)
print(f"Holdout Poisson deviance  GLM={dev('freq_glm'):.4f}  XGB={dev('freq_xgb'):.4f}  "
      f"null={mean_poisson_deviance(test.n_claims, np.full(len(test), train.n_claims.sum()/train.exposure.sum())*test.exposure):.4f}")

# ---- 2. Severity: Gamma GLM (log link) on attritional losses (capped), + separate large-loss load
c = clm.merge(pol, on="policy_id")
c["cap_amt"] = c.claim_amount.clip(upper=LARGE_LOSS_THRESHOLD)
ctr = c[c.policy_id.isin(train.policy_id)]
S = "cap_amt ~ young + vehicle_power + C(region)"
sev = smf.glm(S, ctr, family=sm.families.Gamma(sm.families.links.Log())).fit()
phi = sev.scale                                   # Gamma dispersion -> CV^2
# large-loss load: expected excess over threshold per claim (credibility-free portfolio average)
excess_per_claim = (ctr.claim_amount - LARGE_LOSS_THRESHOLD).clip(lower=0).sum() / len(ctr)

# ---- 3. Pure premium & risk-based technical premium (full portfolio, annualised, using GLM frequency)
pol["freq"] = freq.predict(pol, offset=np.zeros(len(pol)))               # annual frequency (exposure=1)
pol["sev"] = sev.predict(pol) + excess_per_claim
pol["pure_premium_annual"] = pol.freq * pol.sev
pol["pure_premium"] = pol.pure_premium_annual * pol.exposure             # expected loss for the policy period
# Compound-Poisson variance: Var = lambda * E[X^2];  E[X^2] = mu^2 (1+phi) for Gamma; large-loss approximated by sample 2nd moment
ex2_large = ((c.claim_amount.clip(lower=LARGE_LOSS_THRESHOLD) - LARGE_LOSS_THRESHOLD) ** 2).mean()
ex2 = (pol.sev - excess_per_claim) ** 2 * (1 + phi) + ex2_large
pol["var_loss"] = pol.freq * pol.exposure * ex2
pol["std_loss"] = np.sqrt(pol.var_loss)
# std-dev principle at portfolio level (policies diversify), allocated by variance share
pol["risk_load"] = RISK_LOAD_K * np.sqrt(pol.var_loss.sum()) * pol.var_loss / pol.var_loss.sum()
pol["technical_premium"] = (pol.pure_premium + pol.risk_load) / (1 - EXPENSE_RATIO - PROFIT_MARGIN)
pol["rate_adequacy"] = pol.gross_premium / pol.technical_premium             # <1 = underpriced

# ---- save
inc = clm.groupby("policy_id").claim_amount.sum().rename("incurred")
pol = pol.merge(inc, on="policy_id", how="left").fillna({"incurred": 0})
pol.to_csv("../outputs/policy_pricing.csv", index=False)
pd.DataFrame({"coef": freq.params, "p": freq.pvalues}).to_csv("../outputs/freq_glm_coefficients.csv")
pd.DataFrame({"coef": sev.params, "p": sev.pvalues}).to_csv("../outputs/sev_glm_coefficients.csv")
pd.DataFrame({"model": ["Poisson GLM", "XGBoost Poisson"], "holdout_poisson_deviance": [dev("freq_glm"), dev("freq_xgb")]}
             ).to_csv("../outputs/model_comparison.csv", index=False)
print(f"Gamma dispersion={phi:.2f}  large-loss excess/claim={excess_per_claim:,.0f}")
print(f"Portfolio: actual incurred={pol.incurred.sum():,.0f}  model pure premium={pol.pure_premium.sum():,.0f}  "
      f"current premium={pol.gross_premium.sum():,.0f}  technical premium={pol.technical_premium.sum():,.0f}")
print(freq.summary().tables[1])
