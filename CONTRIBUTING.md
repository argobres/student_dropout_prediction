# Contributing

Open an issue describing the problem or experiment. Use a focused branch and pull request; explain the change and its validation. Keep source data immutable and put generated artifacts under reports/generated. Run `python -m pytest -q` before submitting.

Preserve the held-out protocol. Any new modelling decision informed by published test results requires fresh evaluation data for an unbiased performance claim. Document feature availability and cohort timing.

Use concise, factual commit messages such as `fix: preserve undefined semester ratios` or `docs: clarify prediction timing`. Make one coherent change per commit. Do not fabricate historical commits or backdate work.

Do not submit personal student records, credentials, local environment folders or binary models. Clear exploratory notebook outputs before committing changes; keep the frozen final report separate.
