# Contributing

Use a virtual environment and install `.[examples]`. Run
`python -m unittest discover -s tests -v` with CPU execution before submitting a
pull request. Keep changes focused and describe state/gradient/recovery semantics
and the evidence for any performance claim. Include a regression for an observed
correctness bug. Do not add pretrained weights, credentials or task-private data.

Separate engineering fixtures from scientific evaluation. Preserve real negative
results and explicitly report unmeasured functionality. Avoid changing acceptance
thresholds or selecting new evaluation data after observing outcomes.
