-- SQLite / ANSI-style. Load data/policies.csv, data/claims.csv, outputs/policy_pricing.csv as tables.

-- 1. Loss ratio by region and age band
SELECT p.region, p.age_band,
       COUNT(*)                                        AS policies,
       ROUND(SUM(p.exposure), 0)                       AS exposure_years,
       ROUND(SUM(p.gross_premium), 0)                  AS premium,
       ROUND(SUM(COALESCE(c.incurred, 0)), 0)          AS incurred,
       ROUND(SUM(COALESCE(c.incurred, 0)) / SUM(p.gross_premium), 3) AS loss_ratio
FROM policy_pricing p
LEFT JOIN (SELECT policy_id, SUM(claim_amount) AS incurred FROM claims GROUP BY policy_id) c USING (policy_id)
GROUP BY p.region, p.age_band
HAVING SUM(p.gross_premium) > 0
ORDER BY loss_ratio DESC;

-- 2. Large-loss and XoL layer experience (400k xs 100k)
SELECT COUNT(*) AS claims_above_retention,
       ROUND(SUM(MIN(MAX(claim_amount - 100000, 0), 400000)), 0) AS ceded_to_layer,
       ROUND(MAX(claim_amount), 0) AS largest_claim
FROM claims WHERE claim_amount > 100000;

-- 3. Claim frequency & average severity by vehicle power
SELECT vehicle_power,
       ROUND(1.0 * SUM(n_claims) / SUM(exposure), 4)        AS frequency,
       ROUND(SUM(incurred) / NULLIF(SUM(n_claims), 0), 0)    AS avg_severity
FROM policy_pricing
GROUP BY vehicle_power ORDER BY vehicle_power;

-- 4. Under-priced policies (premium below technical premium), ranked by shortfall
SELECT policy_id, region, age_band, gross_premium, ROUND(technical_premium, 2) AS technical_premium,
       ROUND(technical_premium - gross_premium, 2) AS shortfall
FROM policy_pricing WHERE gross_premium < 0.85 * technical_premium
ORDER BY shortfall DESC LIMIT 50;
