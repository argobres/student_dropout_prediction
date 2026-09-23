# Model artifacts

Run `python -m student_dropout.workflow` to generate `final_pipeline.joblib`, containing the fitted pipeline, target encoder, input columns and seed. Binary artifacts are intentionally excluded from Git. The serializable feature function lives in `student_dropout.features`; keep the package and pinned environment available when loading.

Only load joblib files you generated or trust: loading them can execute code. This research model has not been validated for operational use.
