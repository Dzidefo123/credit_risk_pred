# Population- and Regime-Dependent Transport of Vintage-Aware Macroeconomic Features in Mortgage Competing-Risk Models

**v0.3 INTERNAL DRAFT — for hostile manuscript and author review; not an arXiv submission.**

Author names, affiliations and acknowledgments: [AUTHOR REVIEW REQUIRED].

## Abstract

National macroeconomic inputs can distinguish calendar periods without improving risk ranking among mortgages observed under comparable conditions. We study this distinction in a longitudinal Freddie Mac mortgage competing-risk framework using a frozen mortgage-characteristic baseline and a macro-augmented multinomial model. The macro inputs are vintage-aware and revision-aware; exact provider first-release timestamps were not certified. Evaluation combines temporal proper scores, calibration, calendar-stratified discrimination, population comparisons and conditional-entry cumulative incidence. Development joint log loss decreases from 0.09015<!-- Q19: Q0001 --> to 0.08959<!-- Q19: Q0002 -->, whereas the seasoned seen-vintage temporal comparison worsens from 0.089201<!-- Q19: Q0003 --> to 0.105323<!-- Q19: Q0004 -->. Calendar 2020<!-- Q19: Q0005 --> contributes 89.34%<!-- Q19: Q0006 --> of that deterioration. Pooled payoff AUC increases from 0.56543<!-- Q19: Q0007 --> to 0.62587<!-- Q19: Q0008 -->, but between-year comparisons account for 98.19%<!-- Q19: Q0009 --> of the gain. In a separately registered post-hoc closure over 55<!-- Q19: Q0010 --> supported months, pair-weighted within-month AUC is 0.56014<!-- Q19: Q0011 --> versus 0.55974<!-- Q19: Q0012 -->; sparse months are excluded. The joint-loss direction reverses slightly in younger unseen vintages, from 0.070631<!-- Q19: Q0013 --> to 0.069811<!-- Q19: Q0014 -->, while other scores do not uniformly improve. Historical cumulative-incidence discrepancies describe a concentrated entry cohort and overlapping realized macro information, rather than an independent prospective forecasting experiment. Payoff-Brier uncertainty differs under facility and calendar resampling. The observed classification is **MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS**. The findings support separating pooled discrimination, within-period ranking and probability quality when validating macro augmentation. They do not establish universal macro failure or identify a causal mechanism; conditional entry, prior outcome exposure and limited calendar support restrict interpretation.

## 1. Introduction

Mortgage models are often expected to respond to changing economic conditions. National unemployment, rates, house prices and aggregate activity can supply a plausible connection between an individual loan and its environment. Yet an economically meaningful input can improve a pooled score for reasons that do not meet the intended lending or model-risk use. A mortgage model can assign different levels to different periods, thereby improving comparisons between observations from those periods, without substantially improving comparisons between facilities observed at the same time. A probability model must also get magnitudes right; ranking alone does not establish calibrated probabilities.

This paper asks what survives when those distinctions are made explicit. It studies a frozen monthly default/payoff model ladder with origination characteristics, duration and cohort information, and an added national macro block. Development fit is compared with later evaluation in a seasoned surviving population and with a separate unseen-vintage population. The focus is empirical validation: whether an apparent macro increment reflects stable probability estimation, within-period discrimination, or calendar separation that behaves differently across populations. This is not a test of whether macroeconomic information works in every mortgage application.

The research questions are:

- **RQ-A:** Does vintage-aware macro augmentation improve later competing-risk probability quality relative to the mortgage-characteristic baseline?
- **RQ-B:** When pooled payoff discrimination improves, how much arises within comparable calendar strata and how much between them?
- **RQ-C:** Does the direction of the probability comparison transport to origination vintages absent from development?
- **RQ-D:** How does the uncertainty statement change when facilities rather than calendar blocks are resampled?

The questions concern different estimands. A pooled interval AUC compares observations across months as well as within them. A stratified concordance conditions comparisons on a chosen calendar resolution. A proper score assesses probability quality under an explicit population weighting. A cumulative-incidence projection combines monthly probabilities along an information path and an entry-conditioned risk set. Reporting all these objects is more informative than interpreting a favorable AUC as a general improvement in mortgage risk estimation.

The empirical pattern is population- and calendar-dependent. In the seasoned seen-vintage evaluation, most joint-loss deterioration is concentrated in calendar 2020<!-- Q19: Q0015 -->. A year-level pair decomposition attributes almost all pooled payoff-AUC improvement to between-year comparisons. The admitted month-level closure finds a small negative average within-month difference on supported months. In the younger unseen-vintage population the joint-loss point estimate slightly favors macro augmentation, although several cause-specific comparisons continue to favor the baseline. These observations coexist. Their coexistence does not identify a unique causal account linking the AUC gain to the calendar probability error.

The contribution is a traceable empirical model-risk study using established methods. It brings together vintage/revision-aware macro construction, a temporal competing-risk comparison, pooled-versus-stratified ranking, proper scoring, population transport and explicit uncertainty units. It also exposes the limits of those measurements through a public evidence chain, hostile review and correction history. **DISTINCTIVE COMBINATION PLAUSIBLE WITH MATERIAL LIMITATIONS** remains the literature-audit conclusion. Passing software checks does not upgrade that conclusion or certify scientific, commercial or regulatory validity.

## 2. Related work and positioning

### Mortgage termination and economic conditioning

Default and prepayment have long been treated as competing mortgage terminations, including option-based and heterogeneous-exercise formulations [@Deng1996; @Deng2000]. Bayesian Freddie mortgage competing hazards and the recent joint subdistribution/copula approach provide close domain precedents [@Bhattacharya2019; @Bu2026]. Our monthly multinomial model is an established probability construction; it neither invents competing risks nor estimates the latent dependence object of a copula model. The closest published comparison remains incomplete on its exact sample, information clock and evaluation design, so we do not claim those components are absent from prior work.

Macro conditioning and refinancing incentives are also established. Credit survival and multihorizon mortgage studies motivate economic covariates and changing environments [@Bellotti2009; @Breeden2022; @Breeden2023]. Work on exercise costs and burnout motivates the role of heterogeneous surviving borrowers [@Stanton1995]. These precedents support the research question, not a causal explanation for our observed score difference. A seasoned survivor population may have a different payoff response from a younger population; the present design does not identify that response or observe individual refinancing motives.

### Discrimination, probability quality and transport

Proper scoring, calibration and competing-risk assessment address established validation problems [@Gneiting2007; @VanCalster2019; @Heyard2020]. Censoring-aware AUC and prediction-error references distinguish their control and weighting choices [@Blanche2013; @Gerds2012]. Mortgage and lending studies already use multiple metrics and temporal assessment [@Chen2021; @Li2023]. Recent Freddie survival work explicitly studies simulated drift and calibration, with macro conditioning and competing termination identified as possible extensions [@Peng2026]. We contribute a natural-calendar empirical comparison, without claiming to resolve that paper's full research agenda.

Real-time macro vintages have substantial forecasting precedents, and revision-aware credit applications exist outside mortgage termination [@Croushore2001; @Bianchi2026]. Historical mortgage macro scenarios are another related information contract [@Breeden2020]. Our narrower contract reconstructs ALFRED vintage-date availability and revisions; it does not certify exact initial provider-release timestamps. General shift and calibration literature supports separate probability-quality assessment, but it cannot establish the mechanism of a particular mortgage result [@Ovadia2019; @Gama2014].

The potentially distinctive element is the empirical combination of calendar discrimination decomposition, probability evaluation and cross-population transport under an auditable mortgage competing-risk contract. Individual components are established. No priority or first-study claim is made. The practical contribution is showing how the interpretation changes once the pooled metric is decomposed and countervailing population results are disclosed.

## 3. Data, estimands and experimental protocol

### 3.1 Population, source semantics and conditional entry

Freddie Mac longitudinal mortgage histories supply the empirical population. Facilities, not monthly rows, are the loan-level statistical units. Eligible risk intervals are consecutive monthly observations satisfying the frozen research event and ascertainment rules. The adverse endpoint is a research default proxy, not a certified institution-specific default definition. It is the first observed composite adverse event: a numeric source delinquency state from 3<!-- Q19: Q0016 --> through 99<!-- Q19: Q0017 -->, the retained REO state, or a credit-termination code in 02, 03, 09<!-- Q19: Q0018 -->. The numeric states are source delinquency bands, not exact days past due. Verified payoff/maturity uses source code 01<!-- Q19: Q0019 -->; administrative non-credit exits do not themselves establish default. Payoff/maturity termination is the competing event. It must not be equated with voluntary refinancing: source termination coding and the absence of observed borrower motive prevent that interpretation.

Unknown states, gaps, ambiguous event order and administrative exits follow the retained eligibility and censoring rules. They are not silently converted into active no-event rows. A facility enters a horizon comparison only where the relevant observation boundary establishes eligibility. Risk after that boundary is conditional on having survived into the observed risk set; it is not origination-lifetime risk. Older mortgages in the seen-vintage evaluation are therefore selected survivors. Burnout, seasoning and changing composition are plausible reasons populations may behave differently, but they remain hypotheses rather than identified effects.

Loan-age predictors use the scheduled first-payment proxy. That proxy does not supply an exact origination date or reconstruct unobserved exposure. Source origination-vintage labels and age/cohort predictors must consequently be interpreted under their documented definitions. The age-period-cohort clock relation constrains unconstrained coefficient interpretations. Duration bands and cohort restrictions are predictive specification choices, not a causal identification strategy.

### 3.2 Facility roles, temporal separation and populations

The primary seen-vintage labels are 2006, 2008, 2010, 2014<!-- Q19: Q0020 -->. Development covers 2010-09–2017-12<!-- Q19: Q0021 -->; a purge period covers 2018-01–2018-12<!-- Q19: Q0022 -->; temporal evaluation covers 2019-01–2026-02<!-- Q19: Q0023 -->. Facility roles are assigned deterministically and are disjoint between development and evaluation. Calendar separation is therefore combined with facility separation; differences between the samples are not matched-facility attrition. The purge boundary separates the retained design periods but does not establish independence between adjacent national macro observations.

The supplement evaluates labels 2018, 2020, 2022<!-- Q19: Q0024 -->, absent from primary development. Unseen cohort categories use the frozen encoding fallback rather than parameters estimated on those vintages. The populations share an evaluation calendar window but differ in seasoning, entry opportunities, composition and supported durations. Their comparison is informative about population transport under this encoding contract; it is not a controlled experiment that isolates a single population difference.

**Table A. Frozen interval populations.**

| Population | Facilities | Risk intervals | Research defaults | Payoff/maturity events |
| --- | ---: | ---: | ---: | ---: |
| Development | 42,609<!-- Q19: Q0025 --> | 1,568,661<!-- Q19: Q0026 --> | 1,904<!-- Q19: Q0027 --> | 25,822<!-- Q19: Q0028 --> |
| Seen-vintage temporal | 5,619<!-- Q19: Q0029 --> | 248,939<!-- Q19: Q0030 --> | 280<!-- Q19: Q0031 --> | 3,823<!-- Q19: Q0032 --> |
| Unseen-vintage temporal | 17,352<!-- Q19: Q0033 --> | 652,508<!-- Q19: Q0034 --> | 588<!-- Q19: Q0035 --> | 7,323<!-- Q19: Q0036 --> |

These are the model risk populations, not borrower counts or a claim that every original sampled loan contributes to every analysis. Loss averages weight eligible intervals equally. Facilities with longer eligible follow-up contribute more rows, so interval weighting differs from equal facility weighting and can interact with termination. Resampling facilities preserves their observed histories; it does not change the point estimand to an equally weighted facility score. A facility-weighted sensitivity remains unexecuted.

### 3.3 Model ladder and information scope

M0 uses the retained duration/cohort representation. M1 adds original mortgage characteristics, including credit score, leverage, debt burden, balance, coupon, term, purpose and occupancy. M2 adds the frozen national macro feature block: unemployment_level, unemployment_change_3m, treasury_10y_level, mortgage_30y_level, hpi_yoy, cpi_yoy, gdp_qoq<!-- Q19: Q0037 -->. These are the retained transformation names, not a newly selected feature set. Primary loan covariates are origination information; duration and cohort update the clock representation, and macro conditions vary by reporting month. Current delinquency or other dynamic servicing state is deliberately excluded. Current state still contributes to risk-set and event ascertainment; it is not a predictor. These are not full servicing-state forecasting systems, and the macro increment conditional on current loan state has not been tested.

The multinomial specification assigns coherent monthly probabilities to no event, research default and payoff/maturity. Both M1 and M2 were fitted separately on development data under the fixed estimator and training-only preprocessing contract, with L2<!-- Q19: Q0038 --> penalization. Interactions and period fixed effects are excluded from the primary macro specification. The absence of interactions does not make within-month ranking changes impossible. Cause probabilities share a denominator, and refitting can change the original mortgage coefficients. The additive Task 17 erratum withdraws the contrary structural argument; the ranking conclusions below rest on observed decompositions.

For conditional monthly cause probabilities and a prescribed covariate path, event-free survival and cumulative incidence obey

$$S_i(k)=S_i(k-1)\{1-h_{D,i}(k)-h_{P,i}(k)\},\qquad
F_{c,i}(K)=\sum_{k=1}^{K}S_i(k-1)h_{c,i}(k).$$

This known recursion propagates errors in monthly probabilities and changes the risk remaining for competing events [@Austin2016; @Heyard2020]. It is not a causal intervention on actual borrower behavior. Coherence alone does not ensure calibration, correct conditional-entry weighting or valid censoring assumptions.

### 3.4 Vintage- and revision-aware macro information

The candidate source series are national unemployment, Treasury yield, mortgage rate, house prices, consumer prices and real output. The frozen model uses their specified levels and transformations. ALFRED vintage-specific retrieval distinguishes reference periods, retained availability dates, revisions and acquisition time. Information is joined under the retained vintage-date contract instead of substituting today's revised history.

Exact initial provider-release timestamps and release lags were not certified. Throughout this paper, “vintage-aware and revision-aware” refers to retained ALFRED availability information. It does not mean that every provider's first public disclosure time has independently been verified. Mortgage knowledge time has a separate retrospective limitation: the released performance panel can contain revisions unavailable in a historical servicing system. The combined pipeline should not be represented as a fully reconstructed operational information set.

National values are shared across facilities in a reporting month. The development geometry contains 88<!-- Q19: Q0039 --> distinct macro months; repeated loan rows do not create independent national observations, and even distinct months can be serially dependent. Correlation, support-distance and coefficient diagnostics describe the retained data geometry. They do not identify causal macro effects or prove concept drift.

### 3.5 Metrics, ranking decomposition and uncertainty

Joint log loss and cause-specific Brier scores assess the original probabilities. AUC assesses cause-specific discrimination; diagnostic calibration intercepts, slopes and reliability summaries assess probability mapping. Evaluation calibration diagnostics do not replace the probabilities with fitted recalibration. Joint loss is sensitive to the population's event frequencies and weighting, so it is reported beside cause-specific scores and calibration rather than treated as a complete risk summary [@Gneiting2007; @VanCalster2019].

For payoff AUC, a case is a payoff interval and controls include all non-payoff intervals under the frozen monthly metric. The pooled statistic compares intervals from different facilities and months. At a calendar stratum resolution, case-control pairs split into within-stratum and between-stratum pairs. With pooled pair weight $w$, the identity is

$$A_{\mathrm{pooled}}=w A_{\mathrm{within}}+(1-w)A_{\mathrm{between}},\qquad
\Delta A_{\mathrm{pooled}}=w\Delta A_{\mathrm{within}}+(1-w)\Delta A_{\mathrm{between}}.$$

Contribution shares require the pair weights. Dividing a stratum's gain by the pooled gain without that weight is a ratio of gain magnitudes, not its contribution. The year-level decomposition uses the frozen pooled and per-year summaries. A separately registered, already executed post-hoc local closure computes month-stratified AUC on the unchanged prediction arrays. A month needs at least 20<!-- Q19: Q0040 --> cases and 20<!-- Q19: Q0041 --> controls under the retained support rule. Its exact decomposition uses eligible months only; suppressed months' within-pairs are never relabeled between-month pairs. Full-primary pooled AUCs are reconciled separately.

Fixed-model uncertainty is examined under facility resampling and calendar-year blocks. The retained interval convention is Percentile95 paired fixed-model differences<!-- Q19: Q0042 -->; facility and calendar schemes use 1000<!-- Q19: Q0043 --> and 1000<!-- Q19: Q0044 --> resamples, respectively. Facility intervals condition on the realized shared calendar and observed facility composition. Calendar blocks probe period sensitivity, with 8<!-- Q19: Q0045 --> annual blocks including a partial final year. These units answer different questions; neither supplies broad macroeconomic sampling uncertainty or uncertainty from repeating training and model selection. The retained ex-calendar-2020<!-- Q19: Q0046 --> sensitivity uses aggregate year blocks and is not the same calculation as the original interval-array bootstrap.

### 3.6 Conditional-entry cumulative incidence and evidence reuse

For CIF evaluation, the first eligible evaluation interval is the facility landmark. Observed incidence uses the retained Aalen–Johansen competing-event reference; horizon metrics use the documented pooled censoring weights and control definitions. The conditional-independent-censoring assumption remains unverified, and implementation compatibility with conditional entry remains an author-review item. Foundational references support the estimands rather than certify the implementation [@Aalen1978; @Austin2016; @Blanche2013].

Modeled horizon paths use the subsequent historical vintage-aware macro sequence and advance duration deterministically. They use future realized information relative to landmark entry. They are retrospective sequential evaluations, not prospective horizon forecasts issued at entry. Projections can extend beyond an individual facility's realized follow-up; they describe model paths, not repeated independent realizations of borrower exposure.

The temporal evaluation is frozen but non-virgin: earlier support and outcome information had already been inspected. The original model freeze and prediction hashes constrain retuning, but do not erase prior exposure. Later geometry diagnostics, calendar decompositions and the local closure are explicitly post-hoc. The refinancing representation remains **EXPLORATORY_ONLY**. This manuscript introduces no new fit, feature, split, calibration or predictive measurement. Admission of the existing local closure's registration, script and aggregate outputs provides traceability without claiming it was part of the original untouched evaluation.

## 4. Results

### 4.1 Development performance

Development joint loss decreases from 0.09015<!-- Q19: Q0047 --> for M1 to 0.08959<!-- Q19: Q0048 --> for M2. M0 is 0.09171<!-- Q19: Q0049 -->. This is an in-development fit comparison under a larger predictor specification, not a complexity-adjusted generalization gain. The frozen comparison supplies a motivation for examining transport rather than grounds for selecting macro augmentation from development fit alone.

**Table B. Development fit and seen-vintage temporal scores.**

| Model | Development joint loss | Temporal joint loss | Default Brier | Payoff Brier | Default AUC | Payoff AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 0.09171<!-- Q19: Q0050 --> | 0.089261<!-- Q19: Q0051 --> | 0.001124<!-- Q19: Q0052 --> | 0.015141<!-- Q19: Q0053 --> | 0.607431<!-- Q19: Q0054 --> | 0.542069<!-- Q19: Q0055 --> |
| M1 | 0.09015<!-- Q19: Q0056 --> | 0.089201<!-- Q19: Q0057 --> | 0.001223<!-- Q19: Q0058 --> | 0.015127<!-- Q19: Q0059 --> | 0.693508<!-- Q19: Q0060 --> | 0.565426<!-- Q19: Q0061 --> |
| M2 | 0.08959<!-- Q19: Q0062 --> | 0.105323<!-- Q19: Q0063 --> | 0.001272<!-- Q19: Q0064 --> | 0.019825<!-- Q19: Q0065 --> | 0.622831<!-- Q19: Q0066 --> | 0.625868<!-- Q19: Q0067 --> |

### 4.2 Temporal probability performance

In the seen-vintage population, M2 joint loss is 0.105323<!-- Q19: Q0068 --> against M1's 0.089201<!-- Q19: Q0069 -->, a difference of +0.01612<!-- Q19: Q0070 -->. Cause-specific Brier point estimates also worsen. The payoff calibration mean and slope give a corresponding probability concern: observed monthly payoff frequency is 1.536%<!-- Q19: Q0071 -->, M1's mean is 1.138%<!-- Q19: Q0072 -->, and M2's mean is 2.830%<!-- Q19: Q0073 -->. The payoff calibration slope is 0.509<!-- Q19: Q0074 --> for M1 and 0.249<!-- Q19: Q0075 --> for M2. These diagnostics concern the frozen original probabilities and do not represent an applied calibration correction.

M0's temporal joint loss is close to M1's despite substantially weaker default discrimination. This is a reason to inspect the score vector and the event structure rather than infer that near-equal joint scores make models equivalent. No-event prevalence is not itself the fraction of total log loss attributable to no-event observations. The frozen point differences do not measure business loss, regulatory capital or the value of a lending policy.

### 4.3 Calendar concentration of probability deterioration

**Table C. Retained calendar decomposition of M2-minus-M1 joint loss.** Positive differences favor M1; contribution weights use the full seen-vintage interval population.

| Year | Intervals | Weight | M1 joint loss | M2 joint loss | Difference | Contribution |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2019<!-- Q19: Q0076 --> | 62,976<!-- Q19: Q0077 --> | 25.30%<!-- Q19: Q0078 --> | 0.06867<!-- Q19: Q0079 --> | 0.06885<!-- Q19: Q0080 --> | +0.00018<!-- Q19: Q0081 --> | +0.00005<!-- Q19: Q0082 --> |
| 2020<!-- Q19: Q0083 --> | 51,809<!-- Q19: Q0084 --> | 20.81%<!-- Q19: Q0085 --> | 0.13333<!-- Q19: Q0086 --> | 0.20254<!-- Q19: Q0087 --> | +0.06920<!-- Q19: Q0088 --> | +0.01440<!-- Q19: Q0089 --> |
| 2021<!-- Q19: Q0090 --> | 36,397<!-- Q19: Q0091 --> | 14.62%<!-- Q19: Q0092 --> | 0.13513<!-- Q19: Q0093 --> | 0.12905<!-- Q19: Q0094 --> | -0.00608<!-- Q19: Q0095 --> | -0.00089<!-- Q19: Q0096 --> |
| 2022<!-- Q19: Q0097 --> | 28,068<!-- Q19: Q0098 --> | 11.28%<!-- Q19: Q0099 --> | 0.07659<!-- Q19: Q0100 --> | 0.08239<!-- Q19: Q0101 --> | +0.00581<!-- Q19: Q0102 --> | +0.00065<!-- Q19: Q0103 --> |
| 2023<!-- Q19: Q0104 --> | 24,719<!-- Q19: Q0105 --> | 9.93%<!-- Q19: Q0106 --> | 0.05594<!-- Q19: Q0107 --> | 0.06344<!-- Q19: Q0108 --> | +0.00750<!-- Q19: Q0109 --> | +0.00074<!-- Q19: Q0110 --> |
| 2024<!-- Q19: Q0111 --> | 22,138<!-- Q19: Q0112 --> | 8.89%<!-- Q19: Q0113 --> | 0.05269<!-- Q19: Q0114 --> | 0.05888<!-- Q19: Q0115 --> | +0.00619<!-- Q19: Q0116 --> | +0.00055<!-- Q19: Q0117 --> |
| 2025<!-- Q19: Q0118 --> | 19,781<!-- Q19: Q0119 --> | 7.95%<!-- Q19: Q0120 --> | 0.05938<!-- Q19: Q0121 --> | 0.06632<!-- Q19: Q0122 --> | +0.00694<!-- Q19: Q0123 --> | +0.00055<!-- Q19: Q0124 --> |
| 2026<!-- Q19: Q0125 --> | 3,051<!-- Q19: Q0126 --> | 1.23%<!-- Q19: Q0127 --> | 0.05956<!-- Q19: Q0128 --> | 0.06460<!-- Q19: Q0129 --> | +0.00503<!-- Q19: Q0130 --> | +0.00006<!-- Q19: Q0131 --> |

Calendar 2020<!-- Q19: Q0132 --> contributes +0.01440<!-- Q19: Q0133 --> of the overall +0.01612<!-- Q19: Q0134 --> difference: 89.34%<!-- Q19: Q0135 --> of deterioration on 20.81%<!-- Q19: Q0136 --> of evaluation intervals. It is a dominant contribution in this observed calendar, not an estimate of a COVID effect. The point difference reverses in calendar 2021<!-- Q19: Q0137 -->; 6<!-- Q19: Q0138 --> of the 7<!-- Q19: Q0139 --> other years have positive differences. Both the concentration and the remaining direction should be disclosed.

Removing calendar 2020<!-- Q19: Q0140 --> gives a renormalized subgroup difference of +0.00217<!-- Q19: Q0141 -->, corresponding to 2.8%<!-- Q19: Q0142 --> relative to that subgroup's M1 baseline. The full-period relative deterioration is 18.1%<!-- Q19: Q0143 -->. The contribution residual +0.00172<!-- Q19: Q0144 --> retains the full-population denominator and answers a different accounting question; it must not be substituted for the ex-period mean.

The retained aggregate year-block sensitivity interval for the ex-period difference is [-0.00151<!-- Q19: Q0145 -->, +0.00671<!-- Q19: Q0146 -->] and includes zero. Only 7<!-- Q19: Q0147 --> blocks remain, including a partial final year. This is a retained post-hoc sensitivity, not proof of a significant subgroup effect or an independent replication of the original interval-array bootstrap. Task 19 does not regenerate it. The appropriate description is “highly concentrated in calendar 2020<!-- Q19: Q0148 -->,” not disappearance outside the pandemic or an identified causal explanation.

Retained distinct-month diagnostics put 77<!-- Q19: Q0149 --> of 86<!-- Q19: Q0150 --> evaluation months beyond the development-distance reference. The development macro correlation condition number is 69.1<!-- Q19: Q0151 -->. The unemployment/house-price-growth correlation changes from -0.919<!-- Q19: Q0152 --> in development to -0.239<!-- Q19: Q0153 --> in evaluation. These describe shifted predictor geometry and a potentially unstable mapping, not the probability of a causal transport failure. They are post-hoc evidence and cannot separate macro effects from duration, cohort or survivor composition.

### 4.4 Pooled versus within-period payoff discrimination

**Table D. Year-stratified payoff discrimination.**

| Statistic | M1 | M2 | M2-minus-M1 |
| --- | ---: | ---: | ---: |
| Full-primary pooled AUC | 0.56543<!-- Q19: Q0154 --> | 0.62587<!-- Q19: Q0155 --> | +0.06044<!-- Q19: Q0156 --> |
| Within-year pair-weighted AUC | 0.55555<!-- Q19: Q0157 --> | 0.56191<!-- Q19: Q0158 --> | +0.00636<!-- Q19: Q0159 --> |
| Between-year solved AUC | 0.56748<!-- Q19: Q0160 --> | 0.63915<!-- Q19: Q0161 --> | +0.07167<!-- Q19: Q0162 --> |

Within-year pairs have weight 17.195%<!-- Q19: Q0163 -->; between-year pairs have weight 82.805%<!-- Q19: Q0164 -->. The weighted within-year contribution is +0.001093<!-- Q19: Q0165 --> and the between-year contribution +0.059349<!-- Q19: Q0166 -->. They account for 1.81%<!-- Q19: Q0167 --> and 98.19%<!-- Q19: Q0168 --> of the pooled gain, respectively. These are contributions to the pooled difference, not ratios of unweighted stratum gains. The between-year component overwhelmingly dominates that pooled comparison.

Year-level conditioning still permits comparisons across different months. The admitted local closure therefore reports a finer description. Across 55<!-- Q19: Q0169 --> supported months covering 191,938<!-- Q19: Q0170 --> of 248,939<!-- Q19: Q0171 --> primary intervals, pair-weighted within-month AUC is 0.56014<!-- Q19: Q0172 --> for M1 and 0.55974<!-- Q19: Q0173 --> for M2: difference -0.00040<!-- Q19: Q0174 -->. Equal-month-weighted values are 0.54048<!-- Q19: Q0175 --> and 0.53880<!-- Q19: Q0176 -->. Monthly differences are positive in 26<!-- Q19: Q0177 --> months and negative in 29<!-- Q19: Q0178 -->; the negative average does not mean every month's ranking worsens.

The month-level exact decomposition excludes 31<!-- Q19: Q0179 --> sparse months and applies to the eligible-month population only. Its pooled AUCs are 0.56122<!-- Q19: Q0180 --> and 0.60954<!-- Q19: Q0181 -->, which differ from the full-primary values. The within-month weighted contribution is -0.00000789<!-- Q19: Q0182 --> and the between-month contribution +0.04832276<!-- Q19: Q0183 -->; their sum is that eligible population's +0.04831<!-- Q19: Q0184 --> difference. This support-qualified result gives no average within-month discrimination gain for M2, while retaining between-month separation. It does not establish a universal absence of useful facility ranking or a formal equivalence bound.

The year and month decompositions should not be conflated with facility-level horizon AUC. They concern risk-interval pairs under specified calendar conditioning. Shared macro terms can mathematically change rankings in a multinomial model, and the coefficients were refitted. The empirical result supplies the interpretation; an invalid structural argument does not. Annual probability means and spread diagnostics suggest possible accounts of the pooled gain, but cannot uniquely attribute case-control concordance changes to one calendar cell.

### 4.5 Transport to unseen origination vintages

**Table E. Seen-versus-unseen metric comparison.** Each population retains its own interval weighting.

| Metric | Seen M1 | Seen M2 | Unseen M1 | Unseen M2 |
| --- | ---: | ---: | ---: | ---: |
| Joint log loss | 0.089201<!-- Q19: Q0185 --> | 0.105323<!-- Q19: Q0186 --> | 0.070631<!-- Q19: Q0187 --> | 0.069811<!-- Q19: Q0188 --> |
| Default Brier | 0.001223<!-- Q19: Q0189 --> | 0.001272<!-- Q19: Q0190 --> | 0.000899<!-- Q19: Q0191 --> | 0.000900<!-- Q19: Q0192 --> |
| Payoff Brier | 0.015127<!-- Q19: Q0193 --> | 0.019825<!-- Q19: Q0194 --> | 0.011217<!-- Q19: Q0195 --> | 0.011682<!-- Q19: Q0196 --> |
| Default AUC | 0.693508<!-- Q19: Q0197 --> | 0.622831<!-- Q19: Q0198 --> | 0.777935<!-- Q19: Q0199 --> | 0.736496<!-- Q19: Q0200 --> |
| Payoff AUC | 0.565426<!-- Q19: Q0201 --> | 0.625868<!-- Q19: Q0202 --> | 0.552197<!-- Q19: Q0203 --> | 0.737342<!-- Q19: Q0204 --> |

On 17,352<!-- Q19: Q0205 --> facilities and 652,508<!-- Q19: Q0206 --> intervals, M2's joint-loss difference is -0.000819<!-- Q19: Q0207 -->. The primary metric therefore reverses direction relative to the seasoned seen-vintage result. This is a small point improvement in a different research population, not a universal positive macro verdict. Default and payoff Brier still favor M1, as does default AUC. Payoff AUC improves, but that population has not received its own finer calendar-pair decomposition.

The classification **MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS** describes these differing observed comparisons without changing the original frozen primary decision. Unseen vintages have different seasoning, calendar exposure and encoding support. Younger loans, different survivor selection and omitted burnout are candidate explanations; their contributions have not been isolated. A per-year decomposition for the unseen population is not a retained result and is not manufactured in this revision. Neither uniform macro benefit nor uniform macro failure is supported.

### 4.6 Competing-risk cumulative incidence

The local closure reconciles 5,619<!-- Q19: Q0208 --> landmarks and shows concentrated entry: 99.02%<!-- Q19: Q0209 --> enter in 2019-01<!-- Q19: Q0210 -->, the median entry month. The latest entry is 2020-01<!-- Q19: Q0211 -->. For the 24<!-- Q19: Q0212 -->-month comparison, the dominant entry group's historical horizon window is therefore 2019-01<!-- Q19: Q0213 --> through 2020-12<!-- Q19: Q0214 -->, while later entrants use offset windows. These are strongly overlapping pieces of one realized national history, not independent macro-path experiments. All landmark projection windows at the 24<!-- Q19: Q0215 -->-month horizon include calendar 2020<!-- Q19: Q0216 -->; this concerns the model's projected calendar path, not identical realized follow-up for every facility.

**Table F. Conditional-entry CIF comparison across all frozen horizons.** The observed reference is entry-conditioned competing incidence. Model columns are mean historical-path projections.

| Horizon, months | Observed payoff/maturity | M1 payoff/maturity | M2 payoff/maturity | Observed default | M1 default | M2 default |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 12<!-- Q19: Q0217 --> | 0.1330<!-- Q19: Q0218 --> | 0.1385<!-- Q19: Q0219 --> | 0.1222<!-- Q19: Q0220 --> | 0.0061<!-- Q19: Q0221 --> | 0.0060<!-- Q19: Q0222 --> | 0.0064<!-- Q19: Q0223 --> |
| 24<!-- Q19: Q0224 --> | 0.3367<!-- Q19: Q0225 --> | 0.2587<!-- Q19: Q0226 --> | 0.7581<!-- Q19: Q0227 --> | 0.0360<!-- Q19: Q0228 --> | 0.0107<!-- Q19: Q0229 --> | 0.0127<!-- Q19: Q0230 --> |
| 36<!-- Q19: Q0231 --> | 0.5056<!-- Q19: Q0232 --> | 0.3536<!-- Q19: Q0233 --> | 0.8079<!-- Q19: Q0234 --> | 0.0417<!-- Q19: Q0235 --> | 0.0144<!-- Q19: Q0236 --> | 0.0149<!-- Q19: Q0237 --> |
| 60<!-- Q19: Q0238 --> | 0.6094<!-- Q19: Q0239 --> | 0.4860<!-- Q19: Q0240 --> | 0.8187<!-- Q19: Q0241 --> | 0.0467<!-- Q19: Q0242 --> | 0.0193<!-- Q19: Q0243 --> | 0.0163<!-- Q19: Q0244 --> |

At the shortest horizon M1 is closer than M2 to observed mean payoff incidence. The much larger M2 overprediction appears at subsequent horizons. Reporting only the most dramatic horizon would hide both the early-horizon comparison and default underprediction shared by the baselines. These point discrepancies do not certify any model's acceptability or uncertainty interval. No statistical test of equality of the mean CIFs is invented.

CIF and calendar-score findings use substantially the same cohort and realized period. The CIF comparison is not independent corroboration of a separate transport experiment. It illustrates a cumulative consequence of the fitted probability system over overlapping historical information. The recursion explains numerical propagation, but the specific claim that one calendar overprediction uniquely causes the pooled AUC gain remains a **diagnostic hypothesis**. The report does not substitute that hypothesis for the verified decomposition or identify a borrower-response mechanism.

### 4.7 Statistical-unit sensitivity

**Table G. Fixed-model paired intervals for seen-vintage differences.**

| Difference | Point estimate | Facility lower | Facility upper | Calendar lower | Calendar upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| Joint log loss | +0.01612<!-- Q19: Q0245 --> | +0.01540973<!-- Q19: Q0246 --> | +0.01678353<!-- Q19: Q0247 --> | +0.00005326<!-- Q19: Q0248 --> | +0.04150209<!-- Q19: Q0249 --> |
| Payoff Brier | +0.004698<!-- Q19: Q0250 --> | +0.00456248<!-- Q19: Q0251 --> | +0.00482166<!-- Q19: Q0252 --> | -0.00004738<!-- Q19: Q0253 --> | +0.01321485<!-- Q19: Q0254 --> |

Payoff-Brier deterioration is **ROBUST_FACILITY_ONLY** under the retained descriptive classification: its facility interval excludes zero while its calendar interval includes zero. This is not a general robustness claim. The joint-loss calendar interval preserves the point direction but has a lower endpoint close to zero and wide sensitivity under the limited annual support. Exact retained endpoints are reported without replacing them by zero or treating apparent numerical proximity as a separate significance test.

Facility resampling answers uncertainty conditional on the realized calendar and facility composition; calendar blocks probe sensitivity to period composition. Neither establishes broad macroeconomic sampling uncertainty or includes repeating development. These distinctions also govern the ex-period sensitivity and the absence of an unseen-population uncertainty claim. An observed point reversal and a demonstrated robust improvement are different statements.

## 5. Discussion

### Pooled discrimination and comparable-calendar ranking

The most useful validation lesson is that the comparison set matters. Pooled payoff AUC rewards ordering of risk intervals across calendar conditions. That can be useful when the intended task is to distinguish probabilities across times. It cannot by itself establish better ordering among facilities observed under comparable conditions. In this experiment the decomposition assigns almost all pooled gain to between-year pairs, and the supported-month summary supplies no positive average within-month increment. This is an observed, support-qualified result, not a proof that shared predictors cannot change facility ranks.

The distinction has consequences for challenger assessment. If a model's intended use is a within-date lending or servicing comparison, a cross-period ranking gain answers a different question. Calendar-stratified assessment should complement the pooled statistic and specify the eligible comparison population. If the intended use is across-date risk forecasting, the period signal can be relevant, but calibrated magnitude and availability constraints become especially important. Neither interpretation makes AUC useless; each specifies what an AUC improvement does and does not demonstrate.

### Probability quality and temporal regimes

The joint-loss deterioration is highly concentrated in calendar 2020<!-- Q19: Q0255 -->, while its direction does not disappear entirely outside that cell. This makes the average-period headline an incomplete description. Probability assessment should show calendar contribution, event-specific behavior and uncertainty units alongside the overall mean. The later sign reversal in one calendar year also prevents describing the retained specification as uniformly worse throughout every evaluated period.

The observed AUC gain and temporal probability distortion coexist. The year decomposition establishes the former's pair composition; the score and calibration summaries establish the latter's magnitude. A plausible diagnostic account links exaggerated calendar separation to payoff overprediction. It remains a hypothesis because aggregate means do not uniquely identify the individual case-control rankings responsible for the AUC change. No intervention, alternative population control or causal macro identification converts that account into an established explanation.

### Population transport and survivor composition

The unseen-vintage result changes the study-level interpretation. A severe joint-loss disadvantage in a seasoned surviving population does not carry unchanged into younger unseen vintages. Conversely, a slight unseen joint-loss improvement does not establish generally better probability estimates: Brier and default discrimination continue to favor M1. Transport is therefore metric- and population-dependent within the frozen experiment.

This comparison motivates scientific restraint about age, cohort and calendar attribution. Facilities from older source vintages that remain eligible later are selected survivors. Burnout and unobserved borrower behavior may differ across that risk set and newer loans. Frozen encoding and duration support differ too. The evidence documents these interpretation boundaries but does not estimate their separate contributions. A servicing-state conditional comparison or independently registered population design could strengthen a future study; this revision does not add one.

### Uncertainty and model governance

Reporting a narrow facility interval without its conditioning can overstate what has been learned about temporal transport. National inputs are repeated across many rows, but repeating a macro value does not generate independent macro information. Sparse annual blocks also limit what a calendar sensitivity can establish. Model-risk practice should name the statistical unit, the fixed-model assumption, the shared information path and the supported forecast population when interpreting performance intervals.

In IFRS 9 or IRB research contexts, probability quality, horizons and population alignment can matter to downstream risk measurement. This experiment implements neither a production ECL system nor regulatory PD/LGD/EAD validation and makes no compliance claim. It evaluates no lender-specific decision thresholds, approval policy, loss economics or business rollout. Its contribution to governance is methodological transparency: deterministic cohort and role assignment, frozen model and prediction lineage, reuse controls, registered post-hoc analyses, correction ledgers and explicit claims tied to evidence.

The economically motivated refinancing-gap experiment is a secondary exploratory result on already inspected outcomes. Its mixed scoring and calibration evidence does not supply a validated repair for the primary model. A plausible exercise incentive can encode period and selection alongside borrower opportunity; the experiment does not observe refinance motive, current verified coupon or an explicit burnout history. We retain **EXPLORATORY_ONLY** rather than promoting this extension to independent confirmation.

The Task 17 error illustrates why that apparatus matters. An appealing structural explanation did not survive mathematical checking and was withdrawn in an additive erratum. Task 18's contribution-denominator and early-horizon accuracy corrections are preserved as corrections rather than removed from history. The new manuscript treats software verification as a traceability control, not a substitute for statistical review. The remaining implementation, literature and publication questions remain visible.

## 6. Limitations and remaining research

The study observes one realized national macro history and limited calendar blocks. Distinct-month support is not an established count of independent economic observations. Borrower-level dependence and identifiers are unresolved; facility clustering cannot establish borrower independence. The concentration in calendar 2020<!-- Q19: Q0256 --> and the near-synchronized entry cohort restrict generalization to other regimes, trajectories and portfolio compositions. Overlapping historical CIF windows are not independent stress scenarios or entry-time forecasts. Future macro paths were realized information in this retrospective evaluation.

The seasoned seen-vintage sample is conditional on survival to eligibility. Different entry, age, vintage and encoding support in the unseen population prevent a simple causal interpretation of the score reversal. Origination covariates omit current servicing state, dynamic borrower behavior and explicit burnout history. Local economic variation and individual refinancing motives are unobserved in the national representation. Payoff coding includes maturity, and a negligible maturity contribution has not been demonstrated.

The macro contract controls recorded vintages and revisions but does not certify exact provider first-release timestamps or release lags. Mortgage knowledge time is retrospective. Future uncertainty includes more than the facility and year-block schemes used here: development/model-selection uncertainty and broad macro sampling uncertainty are unestimated. The ex-period sensitivity is a retained aggregate resampling approximation with few blocks. No significance statement is attached to the small unseen improvement or the supported-month AUC difference.

The monthly decomposition excludes unsupported months and is not a complete decomposition of all primary observations. Per-year unseen-vintage performance, facility-weighted loss, complexity-adjusted comparisons, support-restricted evaluation and a current-state macro increment remain unexecuted strengthening analyses. Flexible nonlinear challengers and alternative event/maturity distinctions would require their own protocols and may change the scope of the conclusion. Prior evaluation exposure means even a registered new scoring pass is post-hoc, not a virgin confirmation.

Observed-versus-modeled CIF comparison assumes compatibility between conditional entry, the observed estimator, controls and pooled censoring weights. The **CG03 implementation review remains open**. The **CG06 mortgage information-vintage literature gap remains partially resolved**. Full-text comparison with the closest recent joint mortgage work is incomplete. The contribution remains a plausible combination with material limitations, without asserting methodological priority or completeness of the literature search.

Fannie Mae is a **proposed external replication**, with status **DRAFT_NOT_YET_AUTHORIZED**; Task 13A remains unauthorized. No Fannie outcome result, completed external validation or cross-provider replication is claimed. Provider provenance and authorization must be resolved before that extension proceeds.

## 7. Conclusion

The observed result is **MIXED_TEMPORAL_TRANSPORT_ACROSS_POPULATIONS**. Macro augmentation improves development fit and pooled payoff discrimination, but almost all year-level discrimination gain arises between periods. Supported-month summaries give no average within-month improvement. In the seasoned seen-vintage evaluation, probability deterioration is concentrated in calendar 2020<!-- Q19: Q0257 -->; its joint-loss direction reverses slightly in the younger unseen population, with other metrics remaining unfavorable. The original primary negative decision and the broader mixed population description refer to different scopes and can be stated together.

Credit-risk validation should distinguish the pooled comparison set, within-period discrimination, probability calibration, the target population and the unit of uncertainty. This is a bounded empirical lesson from an audited longitudinal mortgage experiment. It does not establish universal macro failure, a regulatory model or a unique causal mechanism for the observed distortion.

## 8. Reproducibility, data availability and author review

Public artifacts include the frozen protocols, source and prediction hashes, aggregate experiment summaries, evidence registries, literature verification, hostile review, correction ledgers and additive local-closure admission. Every quantitative statement in this draft maps to an exact retained artifact field and display format. Prior manuscripts and scientific artifacts remain unchanged. The local closure was registered and executed before this revision; its script and aggregate outputs are admitted here without another array scoring pass, fit or metric generation.

Loan-level source records, private arrays and model artifacts are not distributed as public manuscript data. Authorized access is governed by provider conditions. Reproducing the public arithmetic and verifying source hashes is distinct from reproducing private-data processing or demonstrating prospective deployment. The manuscript is a scientific revision for further review, not a claim that every runtime environment or assumption has passed acceptance.

**[PUBLICATION TERMS REVIEW REQUIRED]** and **[ETHICS AND AUTHOR REVIEW REQUIRED]** remain before public preprint submission. Authors must complete identity, affiliations, acknowledgment, data-use and submission decisions. Remaining scientific review includes conditional-entry/censoring compatibility, the incomplete closest-paper comparison and the support-qualified new framing. Hosted repository CI is tracked separately from manuscript science; its pre-existing failures are not silently fixed by this writing task.

## References

- **Deng1996**. Yongheng Deng, John M. Quigley, Robert Van Order (1996). [Mortgage default and low downpayment loans: The costs of public subsidy](https://www.sciencedirect.com/science/article/pii/0166046295021167). Regional Science and Urban Economics. PEER_REVIEWED.
- **Deng2000**. Yongheng Deng, John M. Quigley, Robert Van Order (2000). [Mortgage Terminations, Heterogeneity and the Exercise of Mortgage Options](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00110). Econometrica. PEER_REVIEWED.
- **Bhattacharya2019**. Arnab Bhattacharya, Simon P. Wilson, Refik Soyer (2019). [A Bayesian approach to modeling mortgage default and prepayment](https://www.tcd.ie/research/profiles/?profile=swilson). European Journal of Operational Research. PEER_REVIEWED.
- **Bu2026**. Lan Bu, Fang Wang, Jingping Yang (2026). [The joint model of default and prepayment for a mortgage loan and its application in mortgage insurance](https://www.sciencedirect.com/science/article/abs/pii/S016766872600017X). Insurance: Mathematics and Economics. PEER_REVIEWED.
- **Li2023**. Zhiyong Li, Aimin Li, Anthony Bellotti, Xiao Yao (2023). [The profitability of online loans: A competing risks analysis on default and prepayment](https://www.sciencedirect.com/science/article/pii/S0377221722006567). European Journal of Operational Research. PEER_REVIEWED.
- **Bellotti2009**. Tony Bellotti, Jonathan Crook (2009). [Credit scoring with macroeconomic variables using survival analysis](https://www.research.ed.ac.uk/en/publications/credit-scoring-with-macroeconomic-variables-using-survival-analys/). Journal of the Operational Research Society. PEER_REVIEWED.
- **Breeden2022**. Joseph L. Breeden, Jonathan Crook (2022). [Multihorizon discrete time survival models](https://www.research.ed.ac.uk/en/publications/multihorizon-discrete-time-survival-models/). Journal of the Operational Research Society. PEER_REVIEWED.
- **Breeden2023**. Joseph L. Breeden, Yevgeniya Leonova (2023). [Stabilizing machine learning models with Age-Period-Cohort inputs for scoring and stress testing](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2023.1195810/full). Frontiers in Applied Mathematics and Statistics. PEER_REVIEWED.
- **Chen2021**. Shunqin Chen, Zhengfeng Guo, Xinlei Zhao (2021). [Predicting mortgage early delinquency with machine learning methods](https://www.sciencedirect.com/science/article/abs/pii/S0377221720306846). European Journal of Operational Research. PEER_REVIEWED.
- **Peng2026**. Jianwei Peng, Stefan Lessmann (2026). [Incorporating data drift to perform survival analysis on credit risk](https://www.tandfonline.com/doi/full/10.1080/01605682.2026.2715043). Journal of the Operational Research Society. PEER_REVIEWED.
- **Bianchi2026**. Daniele Bianchi, Teng Jiao (2026). [Macroeconomic Fundamentals and the Shape of Sovereign Credit Risk](https://www.cambridge.org/core/journals/journal-of-financial-and-quantitative-analysis/article/macroeconomic-fundamentals-and-the-shape-of-sovereign-credit-risk/5BD719BD35E94B78FC540582FD2BE63A). Journal of Financial and Quantitative Analysis. PEER_REVIEWED.
- **Croushore2001**. Dean Croushore, Tom Stark (2001). [A real-time data set for macroeconomists](https://www.sciencedirect.com/science/article/pii/S0304407601000720). Journal of Econometrics. PEER_REVIEWED.
- **Stanton1995**. Richard Stanton (1995). [Rational Prepayment and the Valuation of Mortgage-Backed Securities](https://academic.oup.com/rfs/article-abstract/8/3/677/1598001). Review of Financial Studies. PEER_REVIEWED.
- **Gneiting2007**. Tilmann Gneiting, Adrian E Raftery (2007). [Strictly Proper Scoring Rules, Prediction, and Estimation](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf). Journal of the American Statistical Association. PEER_REVIEWED.
- **Blanche2013**. Paul Blanche, Jean‐François Dartigues, Hélène Jacqmin‐Gadda (2013). [Estimating and comparing time‐dependent areas under receiver operating characteristic curves for censored event times with competing risks](https://onlinelibrary.wiley.com/doi/10.1002/sim.5958/full). Statistics in Medicine. PEER_REVIEWED.
- **Gerds2012**. Thomas A. Gerds, Thomas H. Scheike, Per K. Andersen (2012). [Absolute risk regression for competing risks: interpretation, link functions, and prediction](https://pmc.ncbi.nlm.nih.gov/articles/PMC4547456/). Statistics in Medicine. PEER_REVIEWED.
- **Heyard2020**. Rachel Heyard, Jean‐François Timsit, Leonhard Held, COMBACTE‐MAGNET consortium (2020). [Validation of discrete time‐to‐event prediction models in the presence of competing risks](https://pmc.ncbi.nlm.nih.gov/articles/PMC7217187/). Biometrical Journal. PEER_REVIEWED.
- **Austin2016**. Peter C. Austin, Douglas S. Lee, Jason P. Fine (2016). [Introduction to the Analysis of Survival Data in the Presence of Competing Risks](https://www.ahajournals.org/doi/pdf/10.1161/circulationaha.115.017719). Circulation. PEER_REVIEWED.
- **VanCalster2019**. Ben Van Calster, David J. McLernon, Maarten van Smeden, Laure Wynants, Ewout W. Steyerberg, On behalf of Topic Group ‘Evaluating diagnostic tests and prediction models’ of the STRATOS initiative (2019). [Calibration: the Achilles heel of predictive analytics](https://link.springer.com/article/10.1186/s12916-019-1466-7). BMC Medicine. PEER_REVIEWED.
- **Gama2014**. João Gama, Indrė Žliobaitė, Albert Bifet, Mykola Pechenizkiy, Abdelhamid Bouchachia (2014). [A survey on concept drift adaptation](https://research.aalto.fi/fi/publications/a-survey-on-concept-drift-adaptation/). ACM Computing Surveys. PEER_REVIEWED.
- **Breeden2020**. Joseph L. Breeden, Maxim Vaskouski (2020). [Current expected credit loss procyclicality: it depends on the model](https://www.risk.net/journal-of-credit-risk/7516241/current-expected-credit-loss-procyclicality-it-depends-on-the-model). The Journal of Credit Risk. PEER_REVIEWED.
- **Ovadia2019**. Yaniv Ovadia, Emily Fertig, Jie Ren, Zachary Nado, D. Sculley, Sebastian Nowozin, Joshua Dillon, Balaji Lakshminarayanan, Jasper Snoek (2019). [Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html). Advances in Neural Information Processing Systems. PEER_REVIEWED.
- **Aalen1978**. Odd O. Aalen, Søren Johansen (1978). [An Empirical Transition Matrix for Non-Homogeneous Markov Chains Based on Censored Observations](https://www.jstor.org/stable/4615704). Scandinavian Journal of Statistics. PEER_REVIEWED.
