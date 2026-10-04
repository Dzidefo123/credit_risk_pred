-- Inputs: validated portfolio_accounts, portfolio_history and one-row
-- analytics_parameters(as_of). Keep missing booked accounts explicit.
WITH cutoff AS (
    SELECT CASE WHEN CAST(as_of AS DATE) = last_day(as_of) THEN CAST(as_of AS DATE)
                ELSE CAST(date_trunc('month', as_of) - INTERVAL '1' DAY AS DATE) END AS snapshot_date
    FROM analytics_parameters
)
SELECT a.account_id, a.origination_date, a.age, a.monthly_income, a.is_synthetic,
       c.snapshot_date AS observation_date, h.account_id IS NOT NULL AS snapshot_observed,
       h.state, h.dpd, h.default_flag, h.balance, h.credit_limit, h.utilization, h.months_on_book
FROM portfolio_accounts a CROSS JOIN cutoff c
LEFT JOIN portfolio_history h ON h.account_id = a.account_id AND h.observation_date = c.snapshot_date
WHERE a.origination_date <= c.snapshot_date
ORDER BY a.account_id;
