-- Input portfolio_snapshot from portfolio_snapshot.sql; analytics_parameters
-- includes bad_dpd_threshold and delinquency_dpd_threshold. Unknown exposures
-- remain UNOBSERVED with NULL rates, never assumed closed/current/default.
SELECT CASE WHEN s.snapshot_observed THEN s.state ELSE 'UNOBSERVED' END AS state,
       COUNT(*) AS expected_accounts,
       SUM(CASE WHEN s.snapshot_observed THEN 1 ELSE 0 END) AS observed_accounts,
       COALESCE(SUM(s.balance), 0) AS observed_balance,
       SUM(CASE WHEN s.snapshot_observed AND (s.default_flag OR s.dpd >= p.bad_dpd_threshold)
                THEN 1 ELSE 0 END) AS bad_count,
       SUM(CASE WHEN s.snapshot_observed AND (s.default_flag OR s.dpd >= p.delinquency_dpd_threshold)
                THEN 1 ELSE 0 END) AS delinquent_count,
       1.0 * SUM(CASE WHEN s.snapshot_observed AND (s.default_flag OR s.dpd >= p.bad_dpd_threshold)
                      THEN 1 ELSE 0 END)
       / NULLIF(SUM(CASE WHEN s.snapshot_observed THEN 1 ELSE 0 END), 0) AS observed_bad_rate
FROM portfolio_snapshot s CROSS JOIN analytics_parameters p
GROUP BY CASE WHEN s.snapshot_observed THEN s.state ELSE 'UNOBSERVED' END
ORDER BY state;
