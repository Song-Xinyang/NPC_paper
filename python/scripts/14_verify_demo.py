"""Check the synthetic demo's outputs without asserting manuscript results."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main():
    root = Path(__file__).resolve().parents[2]
    with (root / "config/demo_config.yaml").open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    clinical = pd.read_csv(root / cfg["inputs"]["clinical_csv"])
    outdir = root / cfg["modeling"]["output_dir"]
    scores = pd.read_csv(outdir / "endpoint_scores.csv")
    summary = pd.read_csv(outdir / "score_evaluation_summary.csv")
    endpoints = cfg["modeling"]["endpoint_order"]
    patient_id = cfg["clinical"]["patient_id_col"]
    cohort_col = cfg["clinical"]["cohort_col"]

    assert clinical[patient_id].is_unique and not clinical.empty
    assert scores[patient_id].is_unique
    assert set(scores[patient_id]) == set(clinical[patient_id])
    assert np.isfinite(scores[[f"score_{ep}" for ep in endpoints]].to_numpy()).all()
    cohort_sizes = clinical.groupby(cohort_col).size()
    expected = {(ep, cohort) for ep in endpoints for cohort in cohort_sizes.index}
    assert len(summary) == len(expected)
    assert set(zip(summary["endpoint"], summary["cohort"])) == expected
    assert summary["c_index"].between(0, 1).all()
    assert (summary["n"] == summary["cohort"].map(cohort_sizes)).all()
    for ep, cohort in expected:
        assert (outdir / f"KM_{ep}_{cohort}.png").stat().st_size > 0
    print(f"Verified {len(scores)} synthetic patients and {len(summary)} endpoint/cohort evaluations.")


if __name__ == "__main__":
    main()
