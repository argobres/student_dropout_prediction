# Verification record

Verified locally on 24 September 2026 using Anaconda Python 3.13.9 and the package versions in requirements.txt.

- Full reproduction script completed: EDA, all ten model/representation searches, test evaluation, PDP/ICE, group audit and model serialization.
- Four tests passed: source checksum/schema, undefined ratios with input immutability, target rejection, and undefined-rate handling in fairness aggregation.
- Notebook schemas validated and all notebook code cells compiled. The notebook was not separately executed end-to-end; the complete extracted workflow was executed.
- A separate Python process successfully loaded the saved model and generated predictions.
- Exported the submitted report to a self-contained HTML document.

## Reference versus fresh run

| Metric | Submitted reference | Fresh run |
|---|---:|---:|
| RF CV macro F1 | 0.709566 | 0.710165 |
| RF validation macro F1 | 0.728746 | 0.728963 |
| Test macro F1 | 0.719589 | 0.715760 |
| Test accuracy | 0.761582 | 0.759322 |
| Test balanced accuracy | 0.724424 | 0.719812 |

Both runs select the full-feature random forest with 200 trees, balanced class weights, maximum depth 12 and minimum leaf size 5. Logistic-regression full-feature scores match the reference; several tree-based scores differ slightly. The original fitted model and historical environment lockfile were not supplied. The exact cause of the difference has not been established; exact reproduction of historical tree predictions is not claimed. Newly computed fairness metrics correspond to the fresh model and must not be substituted silently into interpretations of the reference model.

## Repository adjustments

Used dropout_prediction_report-2.ipynb because its code matches the other supplied report and it includes a contents section. Fixed the enrollment-only framing to describe semester-two retrospective classification. Replaced mandatory live dictionary scraping with the supplied reference file. Moved deterministic feature engineering into an importable module so serialized models can load outside the training process. Cleaned the working notebook outputs, preserved the submitted outputs in a report snapshot, and organized generated files separately. Added explicit fairness-summary aggregation with undefined-rate handling.

The full dataset contains three second-semester without-evaluation ratios above one; one occurs in the training partition. These values are retained for investigation.

The repository was published publicly at https://github.com/argobres/student_dropout_prediction with the author's supplied Git identity. See the repository Actions tab for current automated test results; the local checks recorded above are separate from CI.
