# Maintenance and reproducibility

Keep changes focused on their scientific or operational purpose. Run the synthetic
demo when changing Python analysis code or its dependencies. Run the affected
research stage on authorized data when changing MRI, training or single-cell code.
Documentation-only changes need a content review. CI does not enforce document
wording, line counts or a prescribed number of commands.

## What is covered

- `requirements/demo/requirements.in` lists the demo's direct dependencies.
  `requirements.txt` beside it pins the installed Python runtime dependencies
  (pip and build tooling are outside this snapshot).
  The snapshot was installed and the demo was exercised on Windows with Python
  3.12.14. The new CI uses Ubuntu 24.04 and Python 3.12; its first remote run is
  still needed to confirm that platform. OS libraries and Python patch versions
  are outside this package snapshot.
- The demo uses 90 synthetic patients, two-fold CV and five bootstrap iterations.
  It checks finite patient scores, cohort/endpoint coverage, C-indices in [0, 1]
  and generated survival plots. Those checks demonstrate executable code, not
  statistical validity or reproduction of the paper. The small synthetic
  threshold/bootstrap groups emit convergence and ill-conditioned-matrix warnings;
  these remain visible in the logs and should not be treated as research results.
- The full Conda/Python environment and R/Bioconductor environment remain
  **unvalidated and unlocked**. No R installation, MRI processing, model training
  or manuscript reproduction was run for this cleanup. Do not describe a green
  demo check as validation of those components.

## Dependency updates

Dependabot groups monthly demo package updates and monthly official GitHub Action
updates. Actions use full commit SHAs and a read-only token. Every demo update is
installed from the PR's snapshot before running the demo. Review scientific
behavior as well as a passing check; no dependency PR is automatically merged.
The broader research environment requires an explicit, separately validated update.

To refresh the entire demo snapshot, use a new Python 3.12 virtual environment:

```bash
python -m pip install -r requirements/demo/requirements.in
python -m pip check
bash run_all.sh demo
python -m pip freeze > requirements/demo/requirements.txt
```

Keep the input list and frozen environment consistent when adding/removing a
direct dependency. Record the actual Python/OS versions and validation result in
the PR; do not commit generated data, plots, models or logs.

## Capture the complete research environment

On the research machine, use a dedicated environment, install all required
packages, and validate the stages that will be reported. Then capture versions
from that actual environment rather than guessing compatible versions:

```bash
mkdir -p environment-snapshots
conda list --explicit > environment-snapshots/conda-explicit.txt
python -m pip freeze > environment-snapshots/python-freeze.txt
Rscript -e 'install.packages("renv", repos="https://cloud.r-project.org")'
Rscript -e 'options(repos=BiocManager::repositories()); renv::snapshot(type="all", lockfile="environment-snapshots/renv.lock", prompt=FALSE); writeLines(capture.output(sessionInfo()), "environment-snapshots/R-session-info.txt")'
```

`renv::snapshot()` records installed R packages, their sources and versions; it
also supports a project without renv initialization. See the official
[snapshot documentation](https://pkgs.rstudio.com/renv/reference/snapshot.html) and
[Bioconductor guide](https://pkgs.rstudio.com/renv/articles/bioconductor.html).
Restore the snapshots in a fresh environment and rerun the affected analysis
before committing them as a verified research baseline. Record the OS, R/Python
versions, hardware, external tools (including nnU-Net), commit and validation
scope alongside them. Review exported paths/URLs for local or private information.

There is intentionally no claimed R lockfile yet: this cleanup machine has no R
runtime or authorized source data to validate the complete single-cell workflow.
