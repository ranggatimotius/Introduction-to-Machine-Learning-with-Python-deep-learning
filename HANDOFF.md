# Introduction to Machine Learning with Python — agent handoff

Updated 28 October 2026 (Asia/Jakarta). This is a handoff checkpoint, not approval to expand scope or deploy. Read repository instructions and inspect the current branch before making changes.

## Purpose and completed scope
Individual learning/class assignment: reproduce core workflows from all eight chapters of Müller and Guido's Introduction to Machine Learning with Python with original explanations, executable examples and committed outputs. This is an educational repository, not a deployed ML application or a verbatim copy of the book.

## Structure and execution
Each `Chapter_01_...` through `Chapter_08_...` contains `notebook.ipynb`. README maps the chapters; `scripts/build_notebooks.py` is the builder; `tests/test_notebooks.py` checks notebook quality. Use the requirements and recommended Python 3.12 environment from README, open Jupyter/VS Code, or run the documented scripts. Read the full script before regenerating: it may overwrite notebooks and outputs.

## Decisions and next work
Keep Markdown theory, interpretation, limitations and executed examples together. Preserve train/test separation, training-only preprocessing and cross-validation pipelines; don't convert illustrative results into real-world performance claims. Chapter 8 covers deployment concepts, not an implemented production service. No new chapter or model training was authorized by this backup task.

Two sibling projects are physically nested locally but are separate repositories: https://github.com/ranggatimotius/Grokking-Deep-Learning-deep-learning and https://github.com/ranggatimotius/Practical-Statistics-for-Data-Scientists-deep-learning . They are linked in this handoff, not accidental Git submodules. Environments, plotting cache and local dependencies remain excluded.

## Status
All eight chapter notebooks were already published before this documentation sync. Consult their actual saved outputs and tests when continuing. This sync checks documentation/Git state; it does not rerun every training experiment. A new agent should start by asking what learning or assignment revision is next, not regenerate everything automatically.
