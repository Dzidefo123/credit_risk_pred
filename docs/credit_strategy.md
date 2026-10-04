# Credit strategy (Phase 8)

Run from the repository root, using trusted local Phase 4/5 artifacts:

```powershell
.venv\Scripts\python.exe -m credit_risk.cli compare-policies --csv cs-training.csv --run-dir artifacts/phase4-origination-001 --validation-dir artifacts/phase5-validation-001 --config configs/credit_strategy.yaml --output-dir artifacts/phase8-policy-002
```

The runner verifies the source, frozen base code/dependencies and assignments,
selection and selected model hashes, and calibration lock before scoring only the
original development rows. No fitting, recalibration, final-test scoring or policy
champion selection occurs. Hashes detect changes; they do not authenticate an
untrusted pickle. The selected model is the Phase 5 development-preferred XGBoost
with sigmoid calibration. PD estimates the inherited two-year delinquency label,
not independently verified contractual default.

`CreditStrategy` extends the original `DecisionPolicyConfig` without changing its
frozen source. `credit_strategy.yaml` supplies named comparisons; omitted fields
use typed defaults. Names must be unique. All policy and limit assumptions are
written to the output manifest, along with inputs/model/code/artifact hashes.
For prospective scoring, call `decide(raw_applicants, probabilities, policy)` with
one PD per applicant in the same positional order; outcomes are never consulted.

PD strictly below `approve_below_pd` qualifies for automatic approval subject to
proxy checks. PD at or above `decline_at_or_above_pd` declines; the intervening
band receives manual review. Grade G1 contains [0, first upper bound]; later grades
contain (previous bound, upper bound]. Grades and decision thresholds are separate
controls. Every row receives an action and stable reason codes.

Limits use raw monthly income, DebtRatio and revolving utilization. Missing or
zero income, missing debt/utilization or values exceeding configurable automatic
guards route otherwise eligible applicants to review. Higher-PD declines retain
priority. No median income imputation enters policy affordability. Default guards
are debt ratio <= 1 and utilization <= 1; these are illustrative proxy controls.
DebtRatio's source numerator/denominator basis is unverified and cannot support a
claim about actual disposable income or verified debt-service capacity.

For complete inputs within guards:

```
base = min(maximum_limit, MonthlyIncome * income_limit_multiplier)
cap = base / (1 + DebtRatio) / (1 + utilization_penalty * utilization)
cap *= grade_limit_factor * (1 - PD)
cap = floor(cap / limit_increment) * limit_increment
```

Defaults: maximum 10,000; multiplier 2; increment 100; grade factors
[1, .8, .6, .4, .2]. Factors must be nonincreasing and in [0,1]. Caps below the
minimum 500 (or zero) trigger review; they are never rounded up to the minimum.
Only APPROVE receives a recommended limit; review and decline have zero automatic
offer/exposure. `indicative_cap` is a diagnostic for complete supported inputs,
not a promised offer. No review is assumed to convert to a funded loan.

Assumed EAD = offered limit * assumed drawdown (default .5). Loss proxy =
PD * assumed LGD (default .45) * assumed EAD. The horizon follows the inherited
two-year label; no annualization is applied. All money figures use unverified
source income units. They are hypothetical, not balances from the original data,
accounting ECL, validated loss forecasts or a recommended lender policy.

Outputs include per-policy row decisions, aggregate approval/review/decline counts,
approved mean PD and expected bad count, offered limits, EAD, loss proxies and
approved grade composition. Historical selected bad rates describe the same
retrospective development population; they are not future funded outcomes.
No reject outcomes are imputed. Independent policy validation, costs/profit,
capacity constraints, customer response, fairness review and reject inference
remain outside this comparison. Phase 9 addresses reject-inference assumptions.
