-- Inputs: validated portfolio_accounts, portfolio_history; a one-row
-- analytics_parameters(as_of, max_months_on_book, bad_dpd_threshold,
-- delinquency_dpd_threshold), and mob_calendar(months_on_book) integers 0..max.
-- Do not mix synthetic/real sources. Tested with DuckDB; Spark-style date/window
-- expressions, but execution on Databricks is not certified by these tests.
WITH account_grid AS (
    SELECT a.account_id, CAST(date_trunc('month', a.origination_date) AS DATE) AS origination_month,
           m.months_on_book, p.as_of, p.bad_dpd_threshold, p.delinquency_dpd_threshold,
           last_day(a.origination_date + m.months_on_book * INTERVAL '1' MONTH) AS observation_date
    FROM portfolio_accounts a CROSS JOIN mob_calendar m CROSS JOIN analytics_parameters p
    WHERE a.origination_date <= p.as_of AND m.months_on_book BETWEEN 0 AND p.max_months_on_book
), account_history AS (
    SELECT g.account_id, g.origination_month, g.months_on_book, g.observation_date,
           g.observation_date <= g.as_of AS matured,
           COUNT(h.account_id) AS observed_prefix_months,
           MAX(CASE WHEN h.default_flag THEN 1 ELSE 0 END) AS observed_ever_default
    FROM account_grid g LEFT JOIN portfolio_history h
      ON h.account_id = g.account_id AND h.months_on_book <= g.months_on_book
     AND h.observation_date <= g.as_of
    GROUP BY g.account_id, g.origination_month, g.months_on_book, g.observation_date, g.as_of
), cumulative AS (
    SELECT origination_month, months_on_book, observation_date, matured,
           COUNT(*) AS cohort_accounts,
           SUM(CASE WHEN observed_prefix_months = months_on_book + 1 THEN 1 ELSE 0 END) AS complete_history_accounts,
           SUM(observed_ever_default) AS cumulative_observed_default_count
    FROM account_history
    GROUP BY origination_month, months_on_book, observation_date, matured
), snapshots AS (
    SELECT g.origination_month, g.months_on_book, COUNT(h.account_id) AS observed_accounts,
           SUM(CASE WHEN h.default_flag OR h.dpd >= g.bad_dpd_threshold THEN 1 ELSE 0 END) AS bad_count,
           SUM(CASE WHEN h.default_flag OR h.dpd >= g.delinquency_dpd_threshold THEN 1 ELSE 0 END) AS delinquent_count,
           SUM(CASE WHEN h.default_flag THEN 1 ELSE 0 END) AS default_count,
           COALESCE(SUM(h.balance), 0) AS balance_exposure
    FROM account_grid g LEFT JOIN portfolio_history h
      ON h.account_id = g.account_id AND h.months_on_book = g.months_on_book
     AND h.observation_date <= g.as_of
    GROUP BY g.origination_month, g.months_on_book
)
SELECT c.origination_month, c.months_on_book, c.observation_date, c.matured, c.cohort_accounts,
       CASE WHEN c.matured THEN s.observed_accounts END AS observed_accounts,
       CASE WHEN c.matured THEN s.bad_count END AS bad_count,
       CASE WHEN c.matured THEN s.delinquent_count END AS delinquent_count,
       CASE WHEN c.matured THEN s.default_count END AS default_count,
       CASE WHEN c.matured THEN s.balance_exposure END AS balance_exposure,
       CASE WHEN c.matured THEN 1.0 * s.observed_accounts / c.cohort_accounts END AS snapshot_coverage,
       1.0 * s.bad_count / NULLIF(s.observed_accounts, 0) AS bad_rate,
       1.0 * s.delinquent_count / NULLIF(s.observed_accounts, 0) AS delinquency_rate,
       CASE WHEN c.matured THEN c.complete_history_accounts END AS complete_history_accounts,
       CASE WHEN c.matured THEN c.cumulative_observed_default_count END AS cumulative_observed_default_count,
       CASE WHEN c.matured THEN 1.0 * c.cumulative_observed_default_count / c.cohort_accounts END AS cumulative_default_lower_bound,
       CASE WHEN c.matured AND c.complete_history_accounts = c.cohort_accounts
            THEN 1.0 * c.cumulative_observed_default_count / c.cohort_accounts END AS cumulative_default_rate
FROM cumulative c JOIN snapshots s
  ON c.origination_month = s.origination_month AND c.months_on_book = s.months_on_book
ORDER BY c.origination_month, c.months_on_book;
