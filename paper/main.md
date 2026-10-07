# Temporal Transport of Mortgage Competing-Risk Probabilities with Point-in-Time Macroeconomic Information

**v0.1 INTERNAL DRAFT — for author review; not an arXiv submission.**

Author names, affiliations and acknowledgments: [AUTHOR REVIEW REQUIRED].

## Abstract

Longitudinal mortgage risk requires calibrated probabilities for both adverse credit events and competing loan termination. An economically plausible predictor can improve ranking without improving the probabilities needed to interpret cumulative risk. We investigate whether a frozen point-in-time macroeconomic feature set improves temporal competing-risk probability quality relative to a mortgage-characteristic baseline in the evaluated Freddie Mac population. The study uses monthly mortgage histories, a research default proxy, payoff/maturity as a competing event, and a discrete-time multinomial model ladder. Evaluation separates development, a purge period, and a later temporal period; uncertainty is assessed with fixed-model facility resampling and a separate calendar-year sensitivity. Macro augmentation improved development joint log loss from 0.09015<!-- NUM: N0001 --> to 0.08959<!-- NUM: N0002 -->, but temporal log loss deteriorated from 0.08920<!-- NUM: N0003 --> to 0.10532<!-- NUM: N0004 -->. Temporal payoff AUC nevertheless increased from 0.565<!-- NUM: N0005 --> to 0.626<!-- NUM: N0006 -->, while payoff Brier score worsened. In the frozen conditional-entry comparison using historical rolling macro paths, observed payoff cumulative incidence at twenty-four months was 33.7%<!-- NUM: N0007 -->, compared with 25.9%<!-- NUM: N0008 --> for the mortgage baseline and 75.8%<!-- NUM: N0009 --> for macro augmentation. These results demonstrate a design-specific divergence between ranking and temporal probability quality. Interpretation is restricted by conditional entry, retrospective mortgage knowledge time, documented prior outcome exposure, and limited calendar resampling support. Historical rolling-path incidence comparisons are not prospective macro forecasts. The evidence supports joint assessment of competing-event probabilities, proper scoring and temporal calibration; it does not establish a causal explanation or cross-provider transportability.

<!-- CLAIM: MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M1_PAYOFF_BRIER, MACRO_PRIMARY_M2_PAYOFF_BRIER, MACRO_CIF_24_OBSERVED_PAYOFF, MACRO_CIF_24_M1_PAYOFF, MACRO_CIF_24_M2_PAYOFF -->

## 1. Introduction

Mortgage credit risk unfolds through a sequence of monthly states and eventual exits. The practical object is not simply a ranking of mortgages by their probability of distress. A longitudinal model also assigns probabilities to remaining active and to termination without the research default event. Those probabilities determine how many facilities remain exposed to future default. A model that ranks one exit well but assigns it excessive probability can therefore produce a misleading cumulative risk picture even when a familiar discrimination statistic improves. This distinction motivates evaluation of the joint probability system rather than assessment of default ranking alone.

The distinction is particularly relevant when a competing payoff endpoint is common. Once a facility pays off, it no longer contributes future exposure to default in the observed mortgage system. Treating that exit as ordinary censoring instead targets a different quantity: a net default risk under a hypothetical removal of the competing event. That estimand can be useful, but it should not be interchangeable with observed-world default cumulative incidence. The comparison must specify whose risk is being estimated, where follow-up begins, which exits are competing, and which losses of observation are censoring. Otherwise apparently similar risk curves can answer different questions.

<!-- CLAIM: SURV_DESIGN, SURV_60_NAIVE_NET_DEFAULT, SURV_60_DEFAULT_CIF -->

Macroeconomic information is a plausible addition to a mortgage-characteristic model. Labor-market conditions, interest rates, housing prices and aggregate activity can be associated with both credit distress and opportunities to terminate a mortgage. However, economic plausibility does not establish temporal stability. A model learned in an earlier calendar period may encounter different combinations of national conditions and a different surviving mortgage population later. Reliable probability estimation requires more than the availability of relevant variables: the mapping learned from development observations must remain useful in the evaluation setting. The present study tests that requirement under a defined information and validation contract.

<!-- CLAIM: PIT_SERIES, PIT_FEATURES, MACRO_ACTUAL_FEATURES, MACRO_LIMITS -->

The central question is whether the frozen PIT macro feature set improves temporal competing-risk probability quality relative to a mortgage baseline in the evaluated Freddie population. The analysis proceeds from an expanded default-prediction benchmark to a conditional-entry competing-risk foundation and then to a multi-vintage macro model ladder. It evaluates development fit separately from temporal performance. It reports default and payoff discrimination alongside joint log loss, cause-specific Brier scores, calibration diagnostics and cumulative incidence. These measurements are complementary: none is treated as a substitute for the others or as a declaration of production readiness.

<!-- CLAIM: PD_CHAMPION, SURV_DESIGN, MACRO_ACTUAL_FEATURES, MACRO_LEDGER -->

The contribution is an auditable empirical validation study, rather than a claim to invent a new survival estimator. First, it specifies a longitudinal research default proxy and a competing payoff endpoint, including conditional entry and ambiguity handling. Second, it connects vintage-aware national macro information to a frozen temporal evaluation design. Third, it documents development improvement that coexists with temporal deterioration and improved payoff ranking. Fourth, it describes the cumulative-incidence discrepancies associated with the frozen joint probabilities. Finally, it presents clearly labeled post-hoc diagnostics and an exploratory refinancing representation without promoting either to a causal explanation or a validated replacement model. Novelty relative to the broader literature remains an author-review and literature-grounding question.

<!-- CLAIM: SURV_DESIGN, PIT_SUPPORT, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_CIF_24_M2_PAYOFF, DIAG_ASSESSMENT, REFI_DECISION -->

## 2. Related work and citation gaps

Three bodies of literature are required to position the study. Mortgage performance research provides the economic and empirical context for default and payoff. Competing-risk methods provide the distinction between cause-specific event probabilities, event-free survival and cumulative incidence. Validation and probability-calibration research provide the framework for interpreting temporal shift and the coexistence of ranking gains with proper-score deterioration. This internal draft identifies those connections without claiming a complete literature review or asserting that the observed pattern is unprecedented.

[CITATION NEEDED: CG01 | mortgage default prediction and longitudinal mortgage validation]

Mortgage prepayment models often examine incentives linked to the difference between a contract rate and a prevailing mortgage rate, as well as features relevant to transaction costs and selection among surviving loans. The present exploratory experiment uses an original-coupon proxy rather than a verified current contract rate or observed refinancing decision. Its connection to that literature must therefore be framed through measurement and estimand compatibility. A study of voluntary refinancing cannot be directly equated with a provider endpoint that combines payoff and maturity. Literature comparisons should preserve this distinction before comparing predictive performance.

[CITATION NEEDED: CG02 | mortgage payoff, prepayment and refinancing incentive modeling]

Survival and competing-risk literature is needed to ground the nonparametric references, horizon scoring and censoring assumptions used here. The paper uses the implemented monthly Aalen–Johansen logic and cumulative/dynamic discrimination convention. It does not add an alternative estimator retrospectively. Reference selection should explain which controls are included when a competing payoff has occurred, and whether an estimator describes net risk or cumulative incidence. That clarification is more important than collecting citations for method names without matching their assumptions to this implementation.

[CITATION NEEDED: CG03 | Aalen–Johansen, competing-risk estimands and conditional entry]
[CITATION NEEDED: CG04 | IPCW competing-risk discrimination and integrated Brier evaluation]

Research on dataset shift and calibration is needed to distinguish a within-provider temporal evaluation from domain or provider transport. Likewise, PIT macro research is needed to distinguish historical vintage availability from a join to revised current series. The repository already contains documented source and information-time contracts; it does not contain a completed scholarly survey establishing how this study differs from every relevant prior model. A later literature task should include null findings and conflicting evidence, avoid direct comparisons across incompatible targets, and distinguish working-paper from peer-reviewed versions.

[CITATION NEEDED: CG05 | temporal dataset shift and probability calibration in credit risk]
[CITATION NEEDED: CG06 | vintage-aware macroeconomic information in credit-risk modeling]

## 3. Problem formulation

Let facility $i$ enter observation at an eligible monthly boundary $e_i$, conditional on remaining in the research risk set. Let $k$ index consecutive future monthly intervals after that entry, and let $t=e_i+k$ identify the corresponding reporting month. The event indicator is $Y_{it}\in\{0,D,P\}$: no observed endpoint, the research default proxy, or payoff/maturity. An interval is included only where the consecutive monthly observations and frozen event-order rules establish that it belongs in the risk set. Unknown states, observation gaps and administrative exits are handled through the documented ascertainment and censoring rules rather than converted into ordinary active observations.

<!-- CLAIM: SURV_DESIGN -->

For an eligible interval, the model assigns discrete conditional probabilities

$$h_D(t\mid x_{it})=P(Y_{it}=D\mid\text{at risk at }t,x_{it}),\qquad
h_P(t\mid x_{it})=P(Y_{it}=P\mid\text{at risk at }t,x_{it}).$$

These are monthly cause probabilities, not continuous-time hazard rates. The multinomial representation also assigns $h_0=1-h_D-h_P$ to no event. With class logits $\eta_0,\eta_D,\eta_P$, the probabilities follow the fitted softmax representation $h_c=\exp(\eta_c)/\sum_{b\in\{0,D,P\}}\exp(\eta_b)$. Default and payoff probabilities consequently share a normalization; the system is not implemented as unrelated binary models whose probabilities are allowed to sum above one.

<!-- CLAIM: MACRO_ACTUAL_FEATURES, SURV_DESIGN -->

For a specified sequence of covariates, conditional event-free survival begins at $S_i(0)=1$ and is updated as

$$S_i(k)=S_i(k-1)\{1-h_{D,i}(k)-h_{P,i}(k)\},$$

with cumulative incidence

$$F_{D,i}(K)=\sum_{k=1}^{K}S_i(k-1)h_{D,i}(k),\qquad
F_{P,i}(K)=\sum_{k=1}^{K}S_i(k-1)h_{P,i}(k).$$

Within this joint construction, increasing modeled payoff probability reduces the event-free probability carried into later intervals, holding other quantities fixed. That is an algebraic property of the model's probability recursion. It is not a causal claim about how a change in an economic variable alters actual borrower behavior. The paper separates numerical propagation through the fitted system from evidence about the real-world reasons its probabilities differ from observed incidence.

<!-- CLAIM: SURV_DESIGN, DIAG_CIF_COMPONENTS -->

The survival index $k$ should not be confused with the duration predictor. The earlier competing-risk foundation uses time since conditional calendar-window entry, with mortgage age at entry separately describing seasoning. The later macro models use a prespecified duration representation based on the scheduled first-payment proxy and cohort structure. Future horizon integration advances the relevant duration deterministically from its entry value. The first-payment month is not an exact origination date, and the analysis does not manufacture exposure before observation begins. All cumulative-risk statements in this paper retain the conditional-entry interpretation.

<!-- CLAIM: SURV_DESIGN, PIT_APC, MACRO_ACTUAL_FEATURES -->

The information boundary is equally important. For a monthly macro risk interval, the assessment information set uses the preceding month-end and admits macro versions only under the frozen as-of and availability rules. However, a multi-month path needed to integrate incidence may include economic information that arrives after entry. The macro CIF comparisons use historical rolling PIT paths: each interval uses information available at its own assessment date. They are retrospective evaluations of a sequential probability mapping, not forecasts of the entire future macro trajectory from the original entry date. No observed future macro path is relabeled as a forecast.

<!-- CLAIM: PIT_FEATURES, MACRO_ACTUAL_FEATURES, MACRO_CIF_24_SUPPORT, REFI_PRESPEC -->

## 4. Data and cohort construction

### 4.1 Provider, sampling and observation structure

The empirical source is the Freddie Mac Single-Family Loan-Level Standard Dataset under the repository's frozen release-aware research contract. The recovered source cohort contains 140,000<!-- NUM: N0010 --> facilities and 7,741,663<!-- NUM: N0011 --> canonical monthly rows across origination vintages 2006, 2008, 2010, 2014, 2018, 2020 and 2022. These counts describe selected mortgage facilities and their histories; they do not establish an equal number of distinct borrowers. A stable cross-facility borrower identity is unavailable, so links among mortgages held by the same person cannot be reconstructed or used to certify borrower-level independence.

<!-- CLAIM: COHORT_RECOVERY, COHORT_ROWS, COHORT_COMPONENTS, COHORT_HASH -->

Selection is deterministic and identifier-based under the frozen sample construction. The expanded early cohort preserves a nested earlier selection, while the seven-vintage recovery records per-vintage sources, eligible universes, sample hashes and harmonization versions. Performance outcomes do not become an input to the identifier selection rule. Provider anomalies were treated through a versioned quarantine protocol rather than silently rewritten. The paper uses the completed recovery manifest, while retaining the earlier incomplete manifest and its stop decision in the audit trail. This distinction prevents an obsolete engineering boundary from being confused with the final empirical source population.

<!-- CLAIM: PD_SAMPLE, COHORT_COMPONENTS, COHORT_HASH -->

The canonical row count is not the fitted-model sample size. Eligibility, entry, calendar support, facility roles and observed endpoints reduce the source histories to different experiment-specific populations. The paper therefore reports these populations separately. It does not describe the macro results as if every selected mortgage contributed to both training and temporal evaluation. Nor does it equate repeated monthly risk rows with independent units. The exact sample, cohort and risk-set hashes remain available in the frozen machine-readable evidence for an authorized reconstruction with provider access.

<!-- CLAIM: PD_COUNTS, SURV_COUNTS, PIT_COUNTS, MACRO_DESIGN_COUNTS -->

### 4.2 Research events and censoring

The default proxy is the first observed composite adverse mortgage event under the frozen protocol. Qualifying states include numeric delinquency bands from 03 through 99, the REO state RA, or credit-related termination codes 02, 03 and 09. These codes represent the source's monthly state semantics; they are not exact daily days-past-due measurements or an independently validated regulatory default definition. Incident-risk eligibility excludes an already observed qualifying event, and event timing follows the first qualifying reporting/event month without automatic backdating.

<!-- CLAIM: SURV_DESIGN, PD_FEATURES -->

Verified termination code 01 is the competing payoff/maturity endpoint. It ends the facility's risk of a later default in the observed system. Because the source endpoint can combine maturity and payoff, the paper uses “payoff” as shorthand for that documented competing category, rather than asserting that every exit is discretionary prepayment or refinancing. An interest-rate incentive can be a useful predictor of this aggregate endpoint without proving the motivation behind a particular termination. The exploratory experiment therefore does not identify observed refinancing decisions.

<!-- CLAIM: SURV_DESIGN, REFI_PRESPEC, REFI_LIMITS -->

Ambiguous same-month default/payoff order and unresolved terminal-date disagreements are quarantined from the primary labels. Administrative exits, observation end, unknown states and gaps are distinct from payoff. A missing state is not assumed to mean current status, and missing exposure is not filled with an invented run of active months. When a research horizon cannot be ascertained before an administrative or observation interruption, the outcome is treated according to the frozen censoring contract. This handling limits the labeled population and places assumptions on any censoring-adjusted evaluation; it is not evidence that the missingness process is uninformative.

<!-- CLAIM: SURV_DESIGN, MACRO_LIMITS -->

### 4.3 Experiment-specific populations

The expanded twelve-month PD experiment uses 2010-vintage monthly landmarks, rather than the later multi-vintage monthly competing-risk target. Its primary eligible cohort has 1,241,045<!-- NUM: N0012 --> landmarks from 19,590<!-- NUM: N0013 --> facilities, including 7,153<!-- NUM: N0014 --> positive landmarks. Repeated positive landmarks can precede the same endpoint and must not be counted as distinct defaulting mortgages. The temporal evaluation uses 128,812<!-- NUM: N0015 --> landmarks from 2,423<!-- NUM: N0016 --> facilities. This benchmark supplies context for discrimination and calibration; it is not interchangeable with a monthly event probability from the macro models.

<!-- CLAIM: PD_COUNTS -->

The competing-risk foundation also uses the selected 2010 vintage, but constructs conditional entry and consecutive monthly risk intervals. The later macro study uses support-eligible facilities from the seven-vintage cohort and evaluates its primary comparison on vintages represented in development. Unseen vintages are a separate supplementary evaluation with frozen reference restrictions. They are not added to the primary headline simply to enlarge the event count. Table T1 distinguishes these designs and makes clear that the large source population is a sampling frame, not a claim of equally large independent evaluation support.

<!-- CLAIM: SURV_COUNTS, MACRO_DESIGN_COUNTS, MACRO_UNSEEN -->

**Table T1. Frozen experiment populations.** Counts are source-linked facility or repeated-row counts; they are not borrower counts. Endpoint counts for the primary macro evaluation refer to distinct first events within its risk intervals.

| Population | Facilities | Repeated rows | Row definition |
| --- | ---: | ---: | --- |
| Selected seven-vintage source | 140,000<!-- NUM: N0017 --> | 7,741,663<!-- NUM: N0018 --> | Canonical monthly histories |
| Expanded PD temporal evaluation | 2,423<!-- NUM: N0019 --> | 128,812<!-- NUM: N0020 --> | Twelve-month landmarks |
| Competing-risk development | 13,813<!-- NUM: N0021 --> | 471,976<!-- NUM: N0022 --> | Monthly risk intervals |
| Competing-risk temporal evaluation | 2,423<!-- NUM: N0023 --> | 132,409<!-- NUM: N0024 --> | Monthly risk intervals |
| Primary macro development | 42,609<!-- NUM: N0025 --> | 1,568,661<!-- NUM: N0026 --> | Support-eligible risk intervals |
| Primary macro temporal evaluation | 5,619<!-- NUM: N0027 --> | 248,939<!-- NUM: N0028 --> | Seen-vintage risk intervals |

### 4.4 National macroeconomic information

The candidate source series are UNRATE, DGS10, MORTGAGE30US, USSTHPI, CPIAUCSL and GDPC1. They represent national unemployment, a Treasury yield, a market mortgage rate, house prices, consumer prices and real activity under their documented source definitions. Vintage-aware acquisition retained metadata, availability information and observations; the feature system does not merely join today's revised history to earlier mortgage rows. The information contract separates economic reference period, publication or retained availability bounds, revisions, retrieval time and the mortgage assessment date. An apparent historical observation is not admitted solely because its reference period precedes the mortgage row.

<!-- CLAIM: PIT_SERIES, PIT_PROVENANCE, PIT_RELEASES, PIT_GAPS -->

The engineered eligibility concepts are unemployment level, its change over three months, Treasury level, mortgage-rate level, their spread, house-price year-on-year growth, consumer-price year-on-year growth and real-GDP quarter-on-quarter growth. Transformation operands must come from the same permitted as-of information set and satisfy the frozen period and freshness requirements. Series IDs, transforms, units and date rules are distinct pieces of metadata. The presence of a source series in acquisition does not mean that every possible transformation of it entered the fitted model.

<!-- CLAIM: PIT_FEATURES -->

Full-feature support begins in September 2010 and extends through February 2026. The reduced historical sensitivity begins in February 2006 and reaches the same endpoint. The original broader macro-support insufficiency is preserved; the support design is not relabeled as proof that all earlier regimes can be studied with the full feature set. The primary model drops the algebraically redundant spread from its coefficient vector while retaining all eligibility concepts for support definition. A separate rate-representation sensitivity changes the representation under a frozen specification. These restrictions limit identification and feature scope rather than optimize results after temporal evaluation.

<!-- CLAIM: PIT_SUPPORT, PIT_RATES, MACRO_ACTUAL_FEATURES, MACRO_RATE_SENS -->

## 5. Methods and experimental protocol

### 5.1 Benchmarks and model ladder

The expanded PD experiment compares a logistic benchmark with a bounded, regularized XGBoost challenger under frozen development groups, selection rules and calibration criteria. Development ends in December 2014; 2015 landmarks are purged, and temporal assessment starts in January 2016. Internal fit, calibration and selection groups are facility-disjoint but share development calendar coverage. Raw probabilities were retained under the frozen calibration decision rather than applying a post-evaluation correction. The default-only monthly hazard projection is a separate net-risk diagnostic, with frozen covariates; it is not promoted to primary payoff-adjusted annual incidence.

<!-- CLAIM: PD_FEATURES, PD_CHAMPION, PD_HAZARD, PD_LIMITS -->

The macro study uses discrete monthly multinomial logistic models. M0 includes duration and vintage structure. M1 adds the frozen mortgage characteristics: origination credit score, loan-to-value, debt-to-income, principal balance, original interest rate, original term, loan purpose and occupancy. M2 adds unemployment level/change, Treasury and mortgage-rate levels, house-price growth, consumer-price growth and real-activity growth. Mortgage characteristics and national macro information are not allowed to absorb future observed delinquency paths. Duration and vintage restrictions, treatment of unrepresented categories, and fixed regularization are part of the experimental specification.

<!-- CLAIM: MACRO_ACTUAL_FEATURES -->

Original principal balance is transformed with $\log(1+b)$ after the frozen nonnegative check. Numeric imputation, centering and scaling are learned from development risk rows only; missing-value indicators and development-only categorical vocabularies are retained. Unknown evaluation categories follow the fixed reference encoding and are separately counted. The multinomial fits use the frozen regularization and optimizer without class reweighting. Neither temporal outcomes nor diagnostic calibration fits update preprocessing. These restrictions are part of what is being evaluated, so the comparison does not assert that every alternative preprocessing or model family would have the same transport behavior.

<!-- CLAIM: MACRO_ACTUAL_FEATURES -->

**Table T2. Frozen model specifications.** Feature lists are conceptual inputs, not unconstrained causal coefficients. Numerical preprocessing and reference-category details remain in the evidence artifacts.

| Model | Specification | Role |
| --- | --- | --- |
| M0 | Duration and vintage structure | Structural reference |
| M1 | M0 plus frozen mortgage characteristics | Primary mortgage baseline |
| M2 | M1 plus frozen national PIT macro terms | Primary macro comparison |
| RATE | Alternative frozen rate representation | Sensitivity only |
| Reduced models | Earlier supported window and reduced macro concepts | Sensitivity only |
| P1/P2 | Mortgage baseline / asymmetric rate-proxy gap augmentation | Exploratory refinancing comparison |

### 5.2 Temporal separation and information boundaries

Primary macro development covers September 2010–December 2017, with calendar 2018 purged. Temporal evaluation covers January 2019–February 2026. The primary vintage comparison uses 2006, 2008, 2010 and 2014; 2018, 2020 and 2022 belong to the unseen-vintage supplement. Facility roles are disjoint, but temporal separation is not a claim that all evaluation information was previously unknown to researchers. Support and aggregate outcomes had documented prior exposure. The frozen evaluation prevents model retuning after the final predictive comparison, while acknowledging that its history differs from a newly acquired untouched dataset.

<!-- CLAIM: MACRO_ACTUAL_FEATURES, MACRO_LEDGER, MACRO_LIMITS -->

This distinction protects interpretation of the experiment. A frozen model comparison can be informative even where development of the support design used previously inspected counts, but the evidentiary claim must describe that sequence. Subsequent diagnostics operate on the inspected predictions and outcomes. They cannot become independent confirmation merely because they are saved in a separate report. Likewise, a later hypothesis drawn from those diagnostics can be prespecified for its own experiment while still remaining exploratory with respect to the shared evaluation population. The paper keeps all three statuses visible.

<!-- CLAIM: MACRO_LEDGER, DIAG_ASSESSMENT, REFI_PRESPEC, REFI_LIMITS -->

### 5.3 Scoring, discrimination and calibration

For the multinomial target, joint log loss averages the negative logarithm of the predicted probability assigned to the observed class. Cause-specific Brier scores average squared errors between the event indicator and the corresponding predicted probability. These are proper scoring rules for the respective probability objects. AUC instead assesses pairwise ranking of an event against its defined comparison observations. It does not assess whether a probability magnitude is accurate. The expanded binary PD benchmark additionally reports average precision. These metrics address different targets and should not be compared across experiments as if they shared a single scale and population.

<!-- CLAIM: MACRO_ACTUAL_FEATURES, PD_LOGISTIC_AVERAGE_PRECISION, MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M1_PAYOFF_BRIER -->

Calibration summaries include observed event frequency, mean predicted probability, and diagnostic intercept/slope fits. The paper distinguishes calibration-in-the-large from the intercept in a joint intercept-and-slope fit. A slope diagnostic describes the relationship between predictions and outcomes in a specified population; fitting it for evaluation does not mean that those predictions were recalibrated. Calibration tables and reliability summaries therefore remain diagnostics of the frozen probabilities. No evaluation intercept correction or redesigned probability transformation is applied to manufacture a better primary result.

<!-- CLAIM: PD_CALIBRATION_INTERCEPT, PD_CALIBRATION_SLOPE, MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF -->

Horizon assessment uses the frozen monthly nonparametric reference and inverse-probability-of-censoring weighting under pooled independent-censoring assumptions. The cumulative/dynamic discrimination convention includes competing-payoff observations among non-default controls rather than restricting all controls to facilities that survive event-free. The integrated Brier quantity covers the supported frozen horizon range. These choices define the estimand; switching the censoring model or control convention after viewing results would constitute a different analysis. The manuscript reports their limitations rather than attempting a new sensitivity outside the evidence freeze.

<!-- CLAIM: SURV_DESIGN, SURV_IBS, SURV_24_CUMULATIVE_DYNAMIC_AUC -->

### 5.4 Uncertainty and dependence

Monthly intervals and overlapping PD landmarks are repeated observations from facilities. The resampling unit must therefore differ from the row count. The expanded PD study uses facility/loan-cluster resampling of fixed-model metrics, and the competing-risk foundation resamples facilities while reestimating its censoring and nonparametric references within the original draws. The primary macro and refinancing comparisons use paired facility resamples so that both models are evaluated on the same draw. Their intervals describe the frozen comparison, not uncertainty from repeating model development and feature selection.

<!-- CLAIM: PD_PAIRED, SURV_DESIGN, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, REFI_PAIRED -->

The primary macro facility bootstrap uses 1,000<!-- NUM: N0029 --> valid draws on 5,619<!-- NUM: N0030 --> facilities. A separate calendar-year block sensitivity recognizes shared economic conditions; it is not combined with facility intervals as if the two supplied interchangeable independent units. Only eight annual blocks, including a partial final year, support this calendar sensitivity. Its wide intervals are part of the evidence. Neither resampling approach reconstructs unknown cross-facility borrower links, training uncertainty or uncertainty about prospective macro scenarios.

<!-- CLAIM: MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS, MACRO_LIMITS -->

### 5.5 Incidence projections and diagnostic protocol

The earlier structural competing-risk model projects conditional-entry incidence using its specified entry-time features and deterministic duration updates. Its dynamic model provides rolling next-month probabilities; it does not supply a long-horizon dynamic forecast without a future-state distribution. The macro model projections integrate historical rolling PIT economic paths from eligible entry. Horizon-specific calendar support restricts which facilities can contribute where a full path would extend beyond the frozen cutoff. An observed reference and modeled mean must therefore be compared within the documented horizon population, not treated as universal lifetime risk.

<!-- CLAIM: SURV_DESIGN, MACRO_CIF_24_SUPPORT, MACRO_ACTUAL_FEATURES -->

Post-hoc diagnostics examine support, composition, coefficients, score accounting, calibration by calendar period and component substitution. Frozen-coefficient ablations change a contribution for diagnostic purposes without constituting a new selected model. An intercept oracle is fitted and scored on the inspected evaluation sample and is explicitly optimistic. Window-specific diagnostic refits describe association instability in different samples; they do not retroactively replace the primary fit. All reported diagnostic quantities were frozen previously. This manuscript performs no new diagnostic fitting, resampling or empirical calculation.

<!-- CLAIM: DIAG_ASSESSMENT, DIAG_ZERO_UNEMPLOYMENT_LEVEL, DIAG_ORACLE, DIAG_COEFFICIENTS -->

## 6. Results

### 6.1 Expanded default-prediction benchmark

The retained logistic benchmark achieved temporal AUC 0.783<!-- NUM: N0031 -->, average precision 0.1763<!-- NUM: N0032 -->, Brier score 0.007423<!-- NUM: N0033 --> and log loss 0.04233<!-- NUM: N0034 -->. Its mean predicted landmark risk was 0.396%<!-- NUM: N0035 -->, compared with observed frequency 0.832%<!-- NUM: N0036 -->. Calibration-in-the-large was +0.987<!-- NUM: N0037 -->, with slope 0.768<!-- NUM: N0038 -->. Discrimination thus coexisted with meaningful temporal underprediction; it did not justify treating raw landmark probabilities as calibrated forecasts.

<!-- CLAIM: PD_LOGISTIC_ROC_AUC, PD_LOGISTIC_AVERAGE_PRECISION, PD_LOGISTIC_BRIER, PD_LOGISTIC_LOG_LOSS, PD_LOGISTIC_MEAN_PROBABILITY, PD_LOGISTIC_OBSERVED_RATE, PD_CALIBRATION_INTERCEPT, PD_CALIBRATION_SLOPE -->

The prespecified shallow XGBoost challenger produced weaker temporal ranking and proper scores than the retained logistic model. Its failure is specific to that bounded architecture, regularization and selection design; it is not evidence that boosting cannot model mortgage risk. The diagnostic supplement also emphasizes distress-state dependence and the difference between imminent monthly boundary recognition and longer-horizon prediction. Neither a high short-interval ranking score nor a frozen-distress hazard projection is interchangeable with a reliable annual competing-risk estimate.

<!-- CLAIM: PD_CHAMPION, PD_DISTRESS, PD_HAZARD, PD_LIMITS -->

**Table T3. Expanded PD temporal evaluation.** AP denotes average precision; these are overlapping twelve-month landmark outcomes, not the multinomial monthly target.

| Model | AUC | AP | Brier | Log loss |
| --- | ---: | ---: | ---: | ---: |
| Logistic | 0.783<!-- NUM: N0039 --> | 0.1763<!-- NUM: N0040 --> | 0.007423<!-- NUM: N0041 --> | 0.04233<!-- NUM: N0042 --> |
| XGBoost | 0.722<!-- NUM: N0043 --> | 0.0175<!-- NUM: N0044 --> | 0.008264<!-- NUM: N0045 --> | 0.04816<!-- NUM: N0046 --> |

### 6.2 Conditional competing-risk foundation

At the descriptive sixty-month horizon, default-only Kaplan–Meier treatment with payoff censored yielded net default risk 5.21%<!-- NUM: N0047 -->, versus observed competing-risk default incidence 3.10%<!-- NUM: N0048 -->. These are different estimands, not a universal declaration that net-risk estimation is wrong. The comparison is conditional on eligible entry in the evaluation calendar window. It does not describe an entire origination-lifetime population. Furthermore, the structural model lacks sufficient training-duration support for a sixty-month modeled projection; the descriptive nonparametric contrast must not be relabeled as a validated model forecast.

<!-- CLAIM: SURV_60_NAIVE_NET_DEFAULT, SURV_60_DEFAULT_CIF, SURV_DURATION -->

Supported shorter-horizon structural-model metrics are reported in Table T4. Cumulative/dynamic AUC and IPCW Brier assess different aspects of the predicted default CIF. Sparse event support at the shortest horizon and conditional independent-censoring assumptions limit precision and interpretation. The integrated default Brier score over the frozen supported range was 0.00738<!-- NUM: N0049 -->. These results establish a competing-risk research foundation with material limitations, not a complete solved lifetime mortgage probability model.

<!-- CLAIM: SURV_12_CUMULATIVE_DYNAMIC_AUC, SURV_24_CUMULATIVE_DYNAMIC_AUC, SURV_36_CUMULATIVE_DYNAMIC_AUC, SURV_IBS, SURV_DESIGN -->

**Table T4. Structural conditional-entry horizon validation.** Horizons are months after eligible window entry. The sixty-month modeled result is deliberately omitted for insufficient training-time support.

| Horizon | Cumulative/dynamic AUC | IPCW Brier |
| --- | ---: | ---: |
| Twelve months | 0.767<!-- NUM: N0050 --> | 0.00567<!-- NUM: N0051 --> |
| Twenty-four months | 0.745<!-- NUM: N0052 --> | 0.00974<!-- NUM: N0053 --> |
| Thirty-six months | 0.747<!-- NUM: N0054 --> | 0.01311<!-- NUM: N0055 --> |

### 6.3 Development gain and temporal deterioration

Development joint log loss decreased from M0 0.09171<!-- NUM: N0056 --> to M1 0.09015<!-- NUM: N0057 --> and M2 0.08959<!-- NUM: N0058 -->. That progression describes fit in the development population; it is not independent evidence of transport. In temporal evaluation, M1 log loss was 0.08920<!-- NUM: N0059 --> and M2 log loss was 0.10532<!-- NUM: N0060 -->. Thus the macro development improvement did not carry into the evaluated later period under the frozen model comparison.

<!-- CLAIM: MACRO_DEVELOPMENT_M0_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M1_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS -->

The paired M2-minus-M1 temporal log-loss difference was +0.01612<!-- NUM: N0061 -->. Its facility-bootstrap percentile interval was [+0.01541<!-- NUM: N0062 -->, +0.01678<!-- NUM: N0063 -->], and the calendar-year sensitivity interval was [+0.00005<!-- NUM: N0064 -->, +0.04150<!-- NUM: N0065 -->]. Positive differences represent deterioration. Both intervals preserve that direction, but their different widths reflect different conditioning and resampling support. Neither interval includes redoing model development or finding a new feature set.

<!-- CLAIM: MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS -->

**Table T5. Frozen macro development and temporal comparison.** Joint loss targets all event classes; cause-specific AUC and Brier target the indicated cause on monthly risk intervals.

| Model | Development log loss | Temporal log loss | Temporal default AUC | Temporal payoff AUC | Temporal payoff Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 0.09171<!-- NUM: N0066 --> | 0.08926<!-- NUM: N0067 --> | 0.607<!-- NUM: N0068 --> | 0.542<!-- NUM: N0069 --> | 0.015141<!-- NUM: N0070 --> |
| M1 | 0.09015<!-- NUM: N0071 --> | 0.08920<!-- NUM: N0072 --> | 0.694<!-- NUM: N0073 --> | 0.565<!-- NUM: N0074 --> | 0.015127<!-- NUM: N0075 --> |
| M2 | 0.08959<!-- NUM: N0076 --> | 0.10532<!-- NUM: N0077 --> | 0.623<!-- NUM: N0078 --> | 0.626<!-- NUM: N0079 --> | 0.019825<!-- NUM: N0080 --> |

### 6.4 Ranking versus probability quality

Payoff AUC increased from 0.565<!-- NUM: N0081 --> for M1 to 0.626<!-- NUM: N0082 --> for M2. Payoff Brier simultaneously worsened from 0.015127<!-- NUM: N0083 --> to 0.019825<!-- NUM: N0084 -->. This is the central ranking/probability contrast. It is an empirical result in the defined evaluation design, not an assertion that a discrimination gain always accompanies calibration failure or that every macro-conditioned mortgage model will behave this way. Choosing the payoff model by AUC alone would conceal the adverse proper-score comparison.

<!-- CLAIM: MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M1_PAYOFF_BRIER, MACRO_PRIMARY_M2_PAYOFF_BRIER -->

The calibration diagnostics corroborate the probability concern. M1's mean predicted monthly payoff probability was 1.138%<!-- NUM: N0085 -->, compared with observed 1.536%<!-- NUM: N0086 -->. M2's mean was 2.830%<!-- NUM: N0087 -->. The joint calibration slope declined from 0.509<!-- NUM: N0088 --> to 0.249<!-- NUM: N0089 -->. These diagnostics describe the original probabilities; no evaluation recalibration was applied. They also show why a directionally better ranking does not compensate for assigning probabilities that are poorly scaled in the temporal population.

<!-- CLAIM: MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF -->

**Table T6. Primary temporal payoff calibration diagnostics.** Intercepts belong to joint intercept/slope fits; they should not be confused with standalone calibration-in-the-large estimates.

| Model | Observed monthly rate | Mean predicted | Joint intercept | Joint slope |
| --- | ---: | ---: | ---: | ---: |
| M1 | 1.536%<!-- NUM: N0090 --> | 1.138%<!-- NUM: N0091 --> | -1.863<!-- NUM: N0092 --> | 0.509<!-- NUM: N0093 --> |
| M2 | 1.536%<!-- NUM: N0094 --> | 2.830%<!-- NUM: N0095 --> | -3.043<!-- NUM: N0096 --> | 0.249<!-- NUM: N0097 --> |

### 6.5 Cumulative-incidence consequences

In the frozen twenty-four-month historical rolling-path comparison, observed payoff CIF was 33.7%<!-- NUM: N0098 -->, the M1 modeled mean was 25.9%<!-- NUM: N0099 -->, and M2 produced 75.8%<!-- NUM: N0100 -->. The discrepancy is substantially larger than the impression conveyed by an improved payoff AUC alone. It is a conditional-entry, horizon-population-specific incidence comparison, not an entry-time macro forecast or a universal payoff-rate estimate for all selected vintages. The relevant horizon support and censoring assumptions remain part of the result.

<!-- CLAIM: MACRO_CIF_24_OBSERVED_PAYOFF, MACRO_CIF_24_M1_PAYOFF, MACRO_CIF_24_M2_PAYOFF, MACRO_CIF_24_SUPPORT -->

In the fitted joint recursion, excessive payoff probability removes modeled event-free mass that could otherwise contribute to future default incidence. The total default CIF, however, depends on both default and payoff sequences. A poor payoff mapping does not imply that the default CIF must be worse at every horizon: offsetting errors can occur. The frozen component-substitution diagnostics examine this numerical interaction. The paper does not infer real-world causal behavior from changing one fitted component, nor propose the mixed-component curve as a newly validated model. Incidence coherence and observed agreement are distinct requirements.

<!-- CLAIM: DIAG_CIF_COMPONENTS, MACRO_CIF_24_M1_DEFAULT, MACRO_CIF_24_M2_DEFAULT -->

## 7. Post-hoc diagnostic analysis

### 7.1 Support and composition

Task 11 is a **post-hoc diagnostic analysis** of already inspected Task 10 results. The distinct-month multivariate distance check found 77<!-- NUM: N0101 --> of 86<!-- NUM: N0102 --> temporal months outside its development-distance reference. The reference is based on the frozen development macro geometry and its development distance threshold; it is not a probability of causal failure or a universal boundary defining all forms of distribution shift. Monthly and interval-weighted views are kept separate because a national macro observation can recur across many facility rows.

<!-- CLAIM: DIAG_SUPPORT, DIAG_RANGE_UNEMPLOYMENT_LEVEL -->

Unemployment, mortgage-rate and house-price-growth range diagnostics also document support shifts. The multivariate and univariate checks are related but not interchangeable: a combination of values can be atypical even where each marginal value appears familiar, and a univariate violation does not determine how a regularized prediction will behave. These observations justify examining temporal support, but they do not identify the score discrepancy attributable to any individual variable. The development reference itself reflects the selected support window and surviving mortgage population.

<!-- CLAIM: DIAG_RANGE_UNEMPLOYMENT_LEVEL, DIAG_RANGE_MORTGAGE_30Y_LEVEL, DIAG_RANGE_HPI_YOY -->

Mortgage composition and survivor diagnostics document differences in seasoning, vintage and characteristics. Because roles are facility-disjoint, split differences cannot be interpreted as matched-facility attrition. Moreover, the proxy clock relation links period, cohort and age. The documented rank deficiency prevents unrestricted identification of all three components. National macro variables, duration terms and cohort structure occupy a constrained predictive specification; their fitted coefficients do not supply a separable causal decomposition of calendar conditions and mortgage selection.

<!-- CLAIM: DIAG_COMPOSITION, DIAG_SURVIVORS, PIT_APC -->

### 7.2 Payoff mappings and calendar behavior

Frozen score accounting associates M2's payoff Brier deterioration with a much larger prediction variance and mean-bias contribution, alongside a changed covariance with outcomes. This decomposition is numerical error accounting, not causal feature importance. Joint-loss partitions show that probabilities assigned on no-event intervals are important to aggregate deterioration; favorable scoring changes on realized event intervals do not ensure a favorable total score. The diagnostic therefore resists a narrative built solely around the model's ability to recognize observed payoff cases.

<!-- CLAIM: DIAG_PAYOFF_VARIANCE, DIAG_NO_EVENT -->

Calendar calibration illustrates a reversal in the direction of the payoff error. In the inspected 2020 calendar cell, observed monthly payoff frequency was 2.21%<!-- NUM: N0103 -->, while M2's mean prediction was 10.05%<!-- NUM: N0104 -->. In the inspected 2023 cell, observed frequency was 0.87%<!-- NUM: N0105 --> and M2 predicted 0.18%<!-- NUM: N0106 -->. These are diagnostic calendar summaries of different risk sets, not measurements of an identified pandemic treatment. The later underprediction cautions against attributing the entire result to one acute calendar episode.

<!-- CLAIM: DIAG_ANNUAL, DIAG_ASSESSMENT -->

Window-specific coefficient diagnostics show changing signs or magnitudes in payoff associations, with correlated predictors and differing populations limiting interpretation. Frozen-coefficient zeroing checks and component substitutions are consistent with sensitivity to particular mappings; they do not establish that removing a variable would yield a transportable model. The same-sample intercept oracle also leaves substantial residual score deterioration. Its improved in-sample level fit is neither independent evidence of a repair nor a reason to amend the frozen temporal comparison.

<!-- CLAIM: DIAG_COEFFICIENTS, DIAG_ZERO_UNEMPLOYMENT_LEVEL, DIAG_ZERO_UNEMPLOYMENT_CHANGE_3M, DIAG_ORACLE, DIAG_CIF_COMPONENTS -->

### 7.3 Limits of attribution

The combined diagnostics are consistent with support extrapolation, unstable payoff mappings and composition changes as plausible contributors. They do not establish a unique causal mechanism, unrestricted macro identification or an independently validated corrective model. The frozen hypothesis assessment does not support pandemic-only or global-intercept-dominance explanations. Partial support for default mapping instability and macro correlation changes remains weaker than an identified contribution to the primary score difference. These distinctions constrain the discussion and the design of any later experiment.

<!-- CLAIM: DIAG_ASSESSMENT, MECHANISM_1, MECHANISM_2, MECHANISM_3, MECHANISM_4, MECHANISM_5, MECHANISM_6, MECHANISM_7, MECHANISM_8 -->

## 8. Exploratory refinancing-incentive analysis

The refinancing experiment is **EXPLORATORY_ONLY**. Its hypothesis followed the inspected macro validation and diagnostic results; it reuses the relevant frozen development and temporal risk arrays without a new independent validation population. The experiment is nevertheless constrained by its own recorded specification, model ladder and sensitivity choices. Prespecifying an exploratory experiment prevents subsequent discretionary model search, but does not transform a hypothesis generated from the same inspected population into independent confirmation.

<!-- CLAIM: REFI_PRESPEC, REFI_LIMITS, REFI_DECISION -->

The proxy gap is defined by $G_{it}=r_i^{\mathrm{orig}}-m_{t}^{\mathrm{PIT}}$, original mortgage coupon minus the PIT market mortgage rate. The label **ORIGINAL_CONTRACT_RATE_PROXY_GAP** is retained because the original coupon is not a verified current contractual rate. The asymmetric representation uses $G^+=\max(G,0)$ and $G^-=\min(G,0)$. P1 is the frozen mortgage baseline; P2 adds those asymmetric terms. The linear-gap and additional-macro representations are sensitivities, not alternatives selected retrospectively as a successful final model. Actual transaction costs, current eligibility to refinance and the motivation behind payoff are not observed by this proxy.

<!-- CLAIM: REFI_PRESPEC -->

The recorded gap distribution shifts materially between development and evaluation; its interval-weighted mean moves from 1.066<!-- NUM: N0107 --> to 0.014<!-- NUM: N0108 --> percentage points. This is a descriptive contrast in different supported mortgage risk populations, not proof that an individual facility's refinancing opportunity changed by the same amount. Original coupon and the shared market rate can encode duration, selection and calendar structure simultaneously. The economically motivated representation therefore remains subject to the transport and information limitations already present in the baseline design.

<!-- CLAIM: REFI_GAP, REFI_LIMITS -->

Development joint loss improved from P1 0.09015<!-- NUM: N0109 --> to P2 0.08967<!-- NUM: N0110 -->. On the exploratory temporal population, loss worsened from 0.08920<!-- NUM: N0111 --> to 0.09578<!-- NUM: N0112 -->. The facility-paired difference was +0.00658<!-- NUM: N0113 -->, with interval [+0.00580<!-- NUM: N0114 -->, +0.00732<!-- NUM: N0115 -->]. These results do not support a claim that the refinancing representation restores overall temporal probability quality.

<!-- CLAIM: REFI_DEVELOPMENT_P1_JOINT_LOG_LOSS, REFI_DEVELOPMENT_P2_JOINT_LOG_LOSS, REFI_EXPLORATORY_P1_JOINT_LOG_LOSS, REFI_EXPLORATORY_P2_JOINT_LOG_LOSS, REFI_PAIRED -->

Payoff AUC improved, and the point estimate of payoff Brier decreased slightly. However, the payoff Brier difference has different uncertainty under facility and calendar resampling: its calendar-year interval includes zero. The mixed result cannot be summarized as a robust probability improvement by quoting only its AUC gain or the favorable direction of a tiny Brier difference. Calibration, CIF comparisons and default spillover remain part of the frozen mixed exploratory decision. No new threshold for observed refinancing behavior is estimated or asserted.

<!-- CLAIM: REFI_EXPLORATORY_P1_PAYOFF_AUC, REFI_EXPLORATORY_P2_PAYOFF_AUC, REFI_EXPLORATORY_P1_PAYOFF_BRIER, REFI_EXPLORATORY_P2_PAYOFF_BRIER, REFI_PAIRED, REFI_CAL, REFI_CIF, REFI_DECISION -->

**Table T7. Exploratory refinancing comparison.** Fixed models on an already inspected temporal population; no independent validation claim.

| Model | Development joint loss | Exploratory joint loss | Payoff AUC | Payoff Brier |
| --- | ---: | ---: | ---: | ---: |
| P1 | 0.09015<!-- NUM: N0116 --> | 0.08920<!-- NUM: N0117 --> | 0.565<!-- NUM: N0118 --> | 0.0151273<!-- NUM: N0119 --> |
| P2 | 0.08967<!-- NUM: N0120 --> | 0.09578<!-- NUM: N0121 --> | 0.625<!-- NUM: N0122 --> | 0.0151159<!-- NUM: N0123 --> |

## 9. Discussion and model-risk implications

The primary result answers a bounded empirical question. In the evaluated Freddie design, PIT macro augmentation improved development fit but did not improve temporal probability quality. The payoff AUC gain alongside worse Brier and joint loss is not a contradiction between metrics: ranking and magnitude accuracy are different properties. A model can order payoff-prone observations more effectively while assigning probabilities that are too extreme or insufficiently stable across calendar populations. The evidence supports keeping proper scoring and calibration visible whenever a model is being assessed for probability use.

<!-- CLAIM: MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_BRIER -->

The competing-event perspective adds a second layer to that interpretation. Payoff probabilities help determine the modeled event-free mass available to experience later default. They should not be viewed as an ancillary nuisance model whose calibration is irrelevant to credit risk. At the same time, the observed discrepancy does not justify declaring every default-only estimator invalid. A net-risk estimator and an observed competing-risk CIF answer different questions. The issue is choosing and clearly communicating the intended quantity, then evaluating the corresponding probability object under its population and censoring assumptions.

<!-- CLAIM: SURV_60_NAIVE_NET_DEFAULT, SURV_60_DEFAULT_CIF, MACRO_CIF_24_M2_PAYOFF, DIAG_CIF_COMPONENTS -->

A further implication concerns information controls. Vintage-aware macro construction addresses a real source of historical information contamination, but is not a guarantee of temporal stability. The model can use legitimately available economic inputs and still encounter support combinations or outcome mappings that differ from development. Moreover, the mortgage histories remain retrospective provider disclosures with unverified operational knowledge time. The PIT label should therefore attach to the macro information contract, not be expanded into a claim that the entire mortgage decision information set has been reconstructed exactly as it existed historically.

<!-- CLAIM: PIT_PROVENANCE, PIT_FEATURES, PIT_GAPS, MACRO_LIMITS, DIAG_SUPPORT -->

The diagnostics suggest priorities for subsequent research rather than a proved explanation of the observed loss difference. One priority is distinguishing stable predictive structure from support-dependent coefficient mappings. Another is understanding how changing composition and duration restrictions influence the allocation of predictive association between mortgage and macro terms. A third is evaluating conditional payoff calibration by supported calendar regimes. None can be solved by treating the existing oracle or an ablation as a deployable replacement. Any corrective experiment requires a separate design and evidence sequence, particularly after the current evaluation outcomes have been inspected.

<!-- CLAIM: DIAG_ASSESSMENT, DIAG_ORACLE, DIAG_COEFFICIENTS, DIAG_COMPOSITION -->

The refinancing experiment narrows the practical lesson. A borrower-relative economic representation is scientifically motivated, but the available original-coupon proxy is incomplete and its exploratory scoring is mixed. It may improve aspects of payoff ordering without satisfying a joint probability objective. Its value here is to constrain a tempting explanation: replacing broad macro terms with a more specific incentive does not automatically establish restored transport in this population. Evidence of a successful later approach must include the scores and uncertainty relevant to its proposed use, rather than inheriting credibility from the narrative appeal of refinancing economics.

<!-- CLAIM: REFI_PRESPEC, REFI_DECISION, REFI_PAIRED -->

For model-risk practice, the study suggests a layered monitoring and validation perspective. Development metrics should be accompanied by temporal assessment; cause-specific ranking should be accompanied by proper scores and calibration; and coherent incidence curves should be accompanied by observed horizon comparisons. Shared national inputs require calendar-level support diagnostics in addition to facility-level uncertainty. These are implications of this study, not statements of a regulatory requirement. The work neither implements institution-specific monitoring thresholds nor evaluates the business consequences of a lending decision or deployed portfolio intervention.

<!-- CLAIM: MACRO_LIMITS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS, DIAG_SUPPORT -->

## 10. Limitations and generalization boundaries

The empirical evidence is facility-level and provider-specific. A facility identifier does not identify a unique person across mortgages, so dependence among loans associated with the same borrower remains unresolved. Resampling facilities accommodates repeated observations within a facility but cannot reconstruct those unknown links. The selected Freddie cohort also should not be equated with all United States mortgages, general consumer lending or a bank's applicant population. Transport to another provider has not been tested; common economic intuition does not substitute for that missing evidence.

<!-- CLAIM: MACRO_LIMITS, PD_LIMITS, FANNIE_PROPOSED -->

Conditional entry changes the population and the meaning of a horizon. Facilities enter only where observation and protocol conditions establish eligibility and survival to that boundary. Older surviving mortgages can differ from those newly observed, and these differences can interact with calendar conditions. Incidence after conditional entry is not lifetime risk from origination. The scheduled first-payment proxy is not an exact origination clock. The paper does not extrapolate over unobserved pre-entry exposure or reinterpret all selected facilities as having comparable seasoning at the beginning of every experiment.

<!-- CLAIM: SURV_DESIGN, SURV_DURATION, PIT_APC -->

The research default proxy is constructed from monthly delinquency and termination semantics. It is not established as equivalent to a regulatory, contractual or accounting default. Likewise, payoff/maturity is not a directly observed voluntary-refinancing event, and monthly reporting does not establish exact day-level event order. Ambiguity handling and censoring remove observations whose targets cannot be reliably assigned under the contract. The resulting probabilities apply to the specified research risk set. They cannot be transferred directly into regulatory PD, LGD, EAD, impairment staging or a production credit policy.

<!-- CLAIM: SURV_DESIGN, MACRO_LIMITS, REFI_LIMITS -->

Information time remains partly unresolved. The macro pipeline uses vintage-aware data and conservative as-of rules, but the mortgage panel is a current-release retrospective disclosure. Historical corrections, revisions and operational availability of mortgage attributes are not fully reconstructed. National macro variables also omit geographic conditions and household-specific information. A national rate or growth measure shared across many rows does not create many independent economic observations. These limitations restrict both claims of operational realism and interpretation of coefficient precision.

<!-- CLAIM: PIT_GAPS, PIT_PROVENANCE, MACRO_LIMITS -->

Age, period and cohort identification is structurally constrained. The proxy identity $\mathrm{period}=\mathrm{cohort\ proxy}+\mathrm{age\ proxy}$ links the clocks, and the frozen design avoids unrestricted simultaneous coefficient interpretations. The cohort proxy here refers to the scheduled first-payment clock, not a reconstructed exact origination date. Duration bands and cohort reference restrictions are predictive choices, not a causal identification strategy. Effects for unrepresented vintages or unsupported duration categories follow documented encoding restrictions rather than being estimated from nonexistent development observations. The unseen-vintage supplement is consequently informative only with its support qualifications. These restrictions cannot be erased by describing a large interval count or a plausible economic mechanism.

<!-- CLAIM: PIT_APC, PIT_RATES, MACRO_UNSEEN, MACRO_LIMITS -->

Prior exposure limits confirmatory interpretation. Early temporal aggregate outcomes and nested predictive results were inspected before later experiments; the macro support design also had documented exposure to counts. Task 10 remained frozen for its final predictive comparison, but is not a virgin holdout. Task 11 is post-hoc, and Task 12 is exploratory hypothesis development on an inspected temporal population. The manuscript does not reinterpret a written specification or a preserved hash as proof that researchers had never seen the relevant outcomes. Independent evidence remains necessary for stronger confirmation of diagnostic explanations and candidate remedies.

<!-- CLAIM: PD_LIMITS, MACRO_LEDGER, MACRO_LIMITS, DIAG_ASSESSMENT, REFI_LIMITS -->

Uncertainty estimates are conditional and incomplete. Fixed-model bootstrap intervals omit uncertainty from repeating training, preprocessing and model selection. Facility resampling conditions on the realized shared macro path. The calendar sensitivity has limited annual-block support and includes a partial final year. Pooled censoring weights assume conditions that may not hold under informative administrative or observation loss. Calibration point summaries and diagnostic refit coefficients do not have fabricated uncertainty intervals added here. A favorable interval for one score is not evidence that all causes, calendar cells and horizons are reliably modeled.

<!-- CLAIM: MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS, MACRO_LIMITS, PD_CALIBRATION_SLOPE, REFI_PAIRED -->

The macro CIF comparisons condition on subsequently realized rolling PIT information. They evaluate a retrospective sequential mapping rather than prospective macro forecasts available at entry. Horizon eligibility and duration support also differ across comparisons; the sixty-month nonparametric foundation does not validate a structural forecast beyond its training-time support. Apparent agreement in one default curve can coexist with offsetting cause errors. The exploratory original-coupon gap does not establish current refinancing eligibility, transaction costs or a behavioral cutoff. No causal or policy interpretation follows from those limitations.

<!-- CLAIM: MACRO_CIF_24_SUPPORT, SURV_DURATION, DIAG_CIF_COMPONENTS, REFI_PRESPEC, REFI_LIMITS -->

**Table T8. Interpretation boundaries.** These boundaries constrain the paper's claims, rather than offering additional empirical results.

| Evidence object | Permitted interpretation | Unsupported substitution |
| --- | --- | --- |
| Facility histories | Mortgage-level repeated observations | Unique borrowers |
| Default proxy | Defined monthly composite research endpoint | Regulatory default equivalence |
| Payoff endpoint | Payoff/maturity competing exit | Observed refinancing motivation |
| Conditional incidence | Risk after eligible observed entry | Origination-lifetime risk |
| PIT macro inputs | Vintage-aware national information | Fully reconstructed mortgage knowledge time |
| Temporal comparison | Evaluated Freddie calendar transport | Completed cross-provider validation |
| Diagnostics | Post-hoc associations and score accounting | Identified causal explanation |
| Refinancing study | Mixed exploratory evidence | Validated solution |

## 11. Proposed external replication

An external replication using Fannie Mae Single-Family Loan Performance data is proposed to assess a second GSE mortgage population. Source provenance still requires confirmation and the protocol remains **DRAFT_NOT_YET_AUTHORIZED**; Task 13A is unauthorized. No external replication results are included. The detailed source-provenance audit remains in the repository rather than the scientific narrative. Its pending status does not convert the present temporal evidence into a cross-provider finding.

<!-- CLAIM: FANNIE_PROPOSED, FANNIE_PROVENANCE -->

## 12. Reproducibility, availability and privacy

The repository separates public code, schemas, source hashes and frozen aggregate evidence from private provider records and model-development inputs. Manuscript numbers are formatted from the Task 14 claim registry, with claim identifiers retained in the source and a machine-readable numeric binding file. Rendering those existing values is distinct from reproducing an empirical experiment. Reconstructing the underlying study requires appropriate provider-data access and the frozen selection, harmonization, information-time and experiment specifications. Public availability of an aggregate artifact does not establish permission to redistribute the corresponding loan-level data.

<!-- CLAIM: COHORT_HASH, COHORT_COMPONENTS, MACRO_LEDGER, FANNIE_PROVENANCE -->

Preservation manifests and source/model hashes document the research boundary. The internal draft does not load frozen fitted models, consume evaluation ledgers, generate predictions, refit calibration or recompute headline scores. Raw loan records, loan identifiers and account/session information are excluded from the manuscript directory. Public figure placeholders and captions identify their evidence requirements; they do not imply that new borrower-level plots or confidence bands were generated. The supplementary architecture points to existing aggregates where a later authorized formatting pass can proceed without empirical recalculation.

Data access remains subject to documented provider conditions, and publication permissions require separate review. Freddie is the empirical source; Fannie contributes no outcome results to this draft. **[PUBLICATION TERMS REVIEW REQUIRED]** before any public preprint submission, including the use of aggregate counts, metrics, coefficients and figures. This statement is a publication-review flag, not a legal conclusion that publication is either permitted or prohibited. Category selection, author affiliations, acknowledgments and any applicable review requirements also remain author responsibilities.

The analysis uses mortgage-level records without establishing borrower identity or attempting to link records to individuals. The manuscript does not claim that the data are anonymous, that an ethics review has approved the work, or that an institutional review exemption has been obtained. **[ETHICS AND AUTHOR REVIEW REQUIRED]**. Its technical scope is credit-risk research, not an IFRS 9 production model, an IRB model, a regulatory PD/LGD/EAD implementation or a validated bank model. Institutional use would require data, policy and validation evidence absent from this study.

<!-- CLAIM: MACRO_LIMITS, PD_LIMITS, FANNIE_PROVENANCE -->

## 13. Conclusion

In the evaluated Freddie competing-risk design, adding the frozen PIT macro feature set improved development fit but did not improve temporal probability quality. Payoff ranking improved while payoff Brier and joint log loss deteriorated. The historical rolling-path CIF comparison illustrates why a competing-event probability system cannot be assessed by ranking alone. These conclusions describe a particular population, information contract and model specification; they do not establish that macroeconomic inputs inherently fail or that a different provider would reproduce the result.

<!-- CLAIM: MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_JOINT_LOG_LOSS, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_BRIER, MACRO_CIF_24_M2_PAYOFF -->

Post-hoc diagnostics provide bounded evidence about plausible contributors, and the refinancing proxy study remains mixed exploratory evidence. Neither establishes a causal explanation or a validated repair. The supported practical lesson is to evaluate temporal proper scores, calibration and cumulative incidence alongside discrimination, with explicit competing-event and information-time assumptions. External replication remains proposed future work. The next step for this internal manuscript is literature grounding and skeptical scientific review, not another retrospective optimization or immediate submission.

<!-- CLAIM: DIAG_ASSESSMENT, REFI_DECISION, FANNIE_PROPOSED -->

## Figure placeholders and captions

[FIGURE F1 HERE]

**F1 — Research architecture.** Source selection, canonical panel, eligibility, macro information clock, model freeze and temporal evaluation. Distinguish facility roles from calendar blocks and show the retrospective mortgage knowledge-time limitation. Use frozen design metadata; do not plot raw identifiers. <!-- CLAIM: PD_SAMPLE, PIT_SUPPORT, MACRO_LEDGER -->

[FIGURE F2 HERE]

**F2 — Competing-risk state and estimand diagram.** Eligible active state, default proxy, payoff/maturity and separate censoring/ambiguity exits. Show the conditional-entry net-risk versus competing-CIF contrast as descriptive estimates, without supplying unsupported long-horizon model forecasts. <!-- CLAIM: SURV_DESIGN, SURV_60_NAIVE_NET_DEFAULT, SURV_60_DEFAULT_CIF -->

[FIGURE F3 HERE]

**F3 — Development versus temporal scores.** Frozen M0/M1/M2 joint losses and the M2-minus-M1 fixed-model temporal intervals. Keep development fit separate from temporal validation; label facility and calendar resampling independently. <!-- CLAIM: MACRO_DEVELOPMENT_M0_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M1_JOINT_LOG_LOSS, MACRO_DEVELOPMENT_M2_JOINT_LOG_LOSS, MACRO_DELTA_FACILITY_JOINT_LOG_LOSS, MACRO_DELTA_CALENDAR_JOINT_LOG_LOSS -->

[FIGURE F4 HERE]

**F4 — Macro support diagnostics.** Distinct-month development reference and temporal distances/range indicators. The diagnostic threshold is a support reference, not causal identification or a deployable rejection rule. No new PCA or distance calculations. <!-- CLAIM: DIAG_SUPPORT, DIAG_RANGE_UNEMPLOYMENT_LEVEL, DIAG_RANGE_MORTGAGE_30Y_LEVEL, DIAG_RANGE_HPI_YOY -->

[FIGURE F5 HERE]

**F5 — Payoff ranking and calibration.** Frozen temporal AUC, Brier, probability means and calibration summaries, with calendar panels labeled post-hoc. No new bins, fitted calibrators or uncertainty bands. <!-- CLAIM: MACRO_PRIMARY_M1_PAYOFF_AUC, MACRO_PRIMARY_M2_PAYOFF_AUC, MACRO_CAL_M1_PAYOFF, MACRO_CAL_M2_PAYOFF, DIAG_ANNUAL -->

[FIGURE F6 HERE]

**F6 — Conditional cumulative incidence.** Frozen observed and modeled payoff/default horizon summaries with population/support notes. Explicitly label the historical rolling PIT path and distinguish diagnostic component substitution from a selected model. <!-- CLAIM: MACRO_CIF_24_OBSERVED_PAYOFF, MACRO_CIF_24_M1_PAYOFF, MACRO_CIF_24_M2_PAYOFF, DIAG_CIF_COMPONENTS -->

[FIGURE F7 HERE]

**F7 — Exploratory refinancing representation.** Frozen gap-distribution summaries and P1/P2 score contrasts. Label original-coupon proxy, inspected evaluation population and mixed proper-score behavior; do not infer actual refinancing decisions. <!-- CLAIM: REFI_GAP, REFI_PAIRED, REFI_DECISION -->

[FIGURE F8 HERE]

**F8 — Proposed external extension, supplement only.** Freddie temporal evidence to a proposed Fannie design, with provenance and authorization gates pending. No result arrows representing completed external confirmation. <!-- CLAIM: FANNIE_PROPOSED, FANNIE_PROVENANCE -->

## References

No new scholarly references are fabricated. Structured citation placeholders above require a dedicated literature-grounding pass. Method names and provider source descriptions are not substitutes for a verified bibliography. Existing repository source-document references will be reviewed for inclusion alongside scholarly references.

## Appendices — draft architecture

### Appendix A. Data contract and event definitions

[APPENDIX CONTENT PENDING AUTHOR REVIEW] Expand the exact event codes, incident-risk eligibility, ambiguity order, administrative exits and observation gaps from SURV_DESIGN and the frozen research contract. Separate research endpoints from institutional default policy.

### Appendix B. Cohort construction

[APPENDIX CONTENT PENDING AUTHOR REVIEW] Present deterministic selection, sample and risk-set hashes, quarantine policy, experiment population flow and the documented supersession of the incomplete manifest. Preserve facility versus borrower interpretation.

### Appendix C. PIT macro transformations

[APPENDIX CONTENT PENDING AUTHOR REVIEW] List each source series, operand information set, transformation, freshness rule and supported window from PIT_FEATURES/PIT_SUPPORT. Show the excluded redundant primary rate representation without recomputation.

### Appendix D. Additional calibration results

[APPENDIX CONTENT PENDING AUTHOR REVIEW] Format existing reliability and calibration aggregates only. Mark missing uncertainty honestly; no new calibrators, bins or confidence bands.

### Appendix E. Calendar and vintage diagnostics

[APPENDIX CONTENT PENDING AUTHOR REVIEW] Format frozen calendar/vintage summaries and post-hoc geometry. Keep sparse-cell suppression and conditional support qualifications; do not describe diagnostic refits as model replacements.

### Appendix F. Refinancing sensitivities

[APPENDIX CONTENT PENDING AUTHOR REVIEW] Present frozen LINEAR and P3 sensitivities, calendar intervals, duration/vintage cells and CIF comparisons. Entire appendix remains EXPLORATORY_ONLY.

### Appendix G. Reproducibility and governance

[APPENDIX CONTENT PENDING AUTHOR REVIEW] Map manuscript result tokens and claim annotations to public artifacts, source commits, hashes and private-data dependencies. Preserve old ledgers and exact historical results.

### Appendix H. Proposed external replication status

**Table T9 placeholder.** Record only proposed Fannie harmonization and pending provenance/authorization, with no outcome metrics. Protocol DRAFT_NOT_YET_AUTHORIZED; no external result. Detailed source investigations remain in the audit trail.
