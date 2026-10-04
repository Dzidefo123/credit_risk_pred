-- Validated inputs and one-row analytics_parameters(as_of, ...), as above.
-- Supply delinquency_states(state, ordinal) in canonical order, ordinals 0..5.
-- Six by six grid retains unsupported rows as NULL probabilities, not identity.
WITH ordered AS (
    SELECT h.account_id, h.observation_date AS origin_date, h.state AS from_state,
           h.balance AS origin_balance,
           LEAD(h.observation_date) OVER (PARTITION BY h.account_id ORDER BY h.observation_date) AS destination_date,
           LEAD(h.state) OVER (PARTITION BY h.account_id ORDER BY h.observation_date) AS to_state
    FROM portfolio_history h CROSS JOIN analytics_parameters p
    WHERE h.observation_date <= p.as_of
), pairs AS (
    SELECT o.* FROM ordered o CROSS JOIN analytics_parameters p
    WHERE o.destination_date = last_day(o.origin_date + INTERVAL '1' MONTH)
      AND o.destination_date <= p.as_of
), cells AS (
    SELECT s.state AS from_state, d.state AS to_state, s.ordinal AS from_ordinal, d.ordinal AS to_ordinal,
           COUNT(p.account_id) AS pairs, COALESCE(SUM(p.origin_balance), 0) AS origin_balance
    FROM delinquency_states s CROSS JOIN delinquency_states d
    LEFT JOIN pairs p ON p.from_state = s.state AND p.to_state = d.state
    GROUP BY s.state, d.state, s.ordinal, d.ordinal
)
SELECT from_state, to_state, pairs, origin_balance,
       1.0 * pairs / NULLIF(SUM(pairs) OVER (PARTITION BY from_state), 0) AS probability,
       origin_balance / NULLIF(SUM(origin_balance) OVER (PARTITION BY from_state), 0) AS balance_probability,
       to_ordinal > from_ordinal AS roll_forward, to_ordinal < from_ordinal AS roll_back,
       from_ordinal BETWEEN 1 AND 4 AND to_ordinal = 0 AS cure,
       from_state <> 'DEFAULT' AND to_state = 'DEFAULT' AS new_default
FROM cells ORDER BY from_ordinal, to_ordinal;
