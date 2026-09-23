# Model card

**Model:** multiclass random forest; supplied winning configuration: 200 trees, max depth 12, minimum leaf size 5, balanced class weights; full engineered feature pipeline.

**Purpose:** capstone research into academic outcomes and voluntary student support. This is retrospective evidence using semester-two information. Enrollment-only deployment is outside scope.

**Data and evaluation:** UCI 697; 4,424 records from one institution, three recorded classes. Stratified seed-42 train/validation/test split, 5-fold training CV, macro F1 selection. Supplied test macro F1 0.7196, accuracy 0.7616, balanced accuracy 0.7244. Enrolled F1 0.5260 is a material limitation.

**Fairness:** recorded gender, age bins, scholarship, debtor and fee-payment status; gender-by-age intersections. Race unavailable. Financial variables are proxies, not a complete measure of socioeconomic status. Report demographic-parity gaps, selection min/max ratios, TPR/FPR gaps and maximum equalised-odds gaps for each outcome. Denominators and Wilson intervals accompany group rates; sparse comparisons require caution.

**Observed concerns:** dropout recall under age 20 is 54.7%, compared with 87.2% for age 35+. Fee-arrears students have a high false-positive rate, based on a small negative-outcome denominator. Differences do not establish discrimination or causation.

**Explainability:** approved units strongly affect predictions. PDP/ICE interventions may create unrealistic combinations when correlated features are varied. Explanations describe model behavior, not causal effects of an intervention.

**Prohibited application:** autonomous punitive, eligibility or disciplinary decisions. Probabilities are not established as calibrated individual risk estimates. No measured retention lift or ROI. Future work requires timestamp validation, later-cohort evaluation, calibration checks and prospectively evaluated fairness mitigations.
