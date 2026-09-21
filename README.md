# World, Curved: Geometric Latent Priors for Longitudinal MS Forecasting

Code, configurations, seeds, and execution logs supporting:

> M. Savaliya, G.R. Sinha, "World, Curved: Geometric Latent Priors for
> Longitudinal MS Forecasting," *Pattern Recognition Letters* (under review,
> manuscript PRLETTERS-D-26-01738).

This repository accompanies the revised manuscript and is released to satisfy
the reproducibility requests made during peer review: source code, random
seeds, model/training configurations, fold-construction code, and raw
execution logs for all reported runs.

## What is in this repository

| File | Contents |
|---|---|
| `prl-world-scl.ipynb` | Main notebook. Defines the SCL flow module, the four world-model variants (Baseline, SCL-Latent, CurvatureDynamics, and the two exploratory variants SCL-Input / SCL-Learnable), the training loop, the primary-split experiment, the 5-seed multi-seed sweep, and the 5-fold × 5-seed cross-validated sweep. |
| `prl-world-scl.log` | Raw stdout/execution log from running `prl-world-scl.ipynb` end to end, including per-run PSNR/SSIM/Dice for every (fold, seed) combination, the hyperparameter grid search, and timing. |
| `prl-world-scl-rev.ipynb` | Comparator notebook. Trains **DirectUNet**, the plain image-to-image forecaster with no latent-dynamics module, through the identical data pipeline, patient split, seed list, and fold membership as the models above. |
| `prl-world-scl-rev.log` | Raw execution log for the DirectUNet notebook (primary split, 5-seed sweep, and 5-fold × 5-seed sweep). |
| `README.md` | This file. |

Per-run results (PSNR / SSIM / Dice for all 100 trainings: 4 models × 5 folds
× 5 seeds) are recoverable in full from the two `.log` files.

## Environment

- Python 3.x, PyTorch (GPU run on a single Kaggle T4/P100-class accelerator)
- Standard scientific stack: numpy, scipy, pandas, scikit-learn
- No non-standard packages; the SCL flow module is implemented from scratch
  using `torch.nn.functional.conv2d` (see the notebook, "SCL Flow" section)

Exact package versions are pinned by the Kaggle notebook environment at
execution time; see the first cell of each notebook for the environment
snapshot printed at run start.

## Data

Longitudinal MR_MS Dataset (Kaggle):
<https://www.kaggle.com/datasets/farahmo/longitudinal-mri-data>

The dataset is third-party and not redistributed in this repository. Both
notebooks expect it mounted at the standard Kaggle input path
(`/kaggle/input/longitudinal-mri-data/...`); adjust the path constant near
the top of each notebook if running elsewhere.

## Reproducing the results

1. Download the dataset above and mount/place it as described.
2. Run `prl-world-scl.ipynb` top to bottom. This reproduces, in order:
   - the primary 70/15/15 patient-level split and single-run ablation table
     (Baseline, SCL-Latent, SCL-Input, CurvatureDynamics, SCL-Learnable);
   - the 5-seed multi-seed sweep on the primary split;
   - the 5-fold × 5-seed cross-validated sweep (25 runs) for Baseline,
     SCL-Latent, and CurvatureDynamics.
3. Run `prl-world-scl-rev.ipynb` top to bottom to reproduce the DirectUNet
   comparator under the same three protocols.

Both notebooks use the same fold-construction call,
`KFold(n_splits=5, shuffle=True, random_state=42)`, applied at the
**patient** level (not slice level), and the same seed list
`{42, 43, 44, 45, 46}`, so fold membership and seed assignment are identical
across all four models and results are paired on (fold, seed).

### Regenerating the parsed results table

The per-(fold, seed) PSNR/SSIM/Dice values used for the variance-decomposition
tables in the paper can be parsed directly from the log files, e.g.:

```python
import re, pandas as pd

def parse_kfold_block(log_path, model_name, start_marker, n_lines=60):
    rows = []
    lines = open(log_path).read().split("\n")
    start = next(i for i, l in enumerate(lines) if start_marker in l)
    for l in lines[start + 1 : start + 1 + n_lines]:
        s = re.sub(r"^\S+s \d+ ", "", l).split()
        if len(s) >= 7 and s[0].isdigit():
            rows.append(dict(model=model_name, fold=int(s[2]), seed=int(s[3]),
                              psnr=float(s[4]), ssim=float(s[5]), dice=float(s[6])))
    return pd.DataFrame(rows)
```

Adjust `start_marker` to the relevant `"Per-fold-per-seed results"` header in
each log; see the logs' table-of-contents-style section headers (grep for
`"Across-fold"`, `"Per-seed"`, `"Primary-split"`) to locate each block.

## Random seeds and determinism

Seeds `{42, 43, 44, 45, 46}` are used for the 5-seed and 5-fold × 5-seed
sweeps; seed `42` alone is used for the primary single-run split.
`torch.backends.cudnn.deterministic = True` and
`torch.backends.cudnn.benchmark = False` are set at the start of each
notebook. Full bitwise determinism is **not** guaranteed on top of this,
because mixed-precision (AMP) reductions on GPU are not bitwise
order-independent; this is why the paper reports seed-to-seed variance
rather than assuming exact reproducibility of any single run's numbers to
the last decimal. Re-running a given (fold, seed) combination should
reproduce results within a small numerical tolerance.

## Correspondence between code and paper sections

| Manuscript section | Notebook location |
|---|---|
| Sec. 4, discrete SCL operator (`u`, `R[u]`, five-point stencil) | `prl-world-scl.ipynb`, "SCL Flow" class definition |
| Sec. 5, WorldModelBase / SCL-Latent / CurvatureDynamics / SCL-Input / SCL-Learnable | `prl-world-scl.ipynb`, model class definitions |
| Sec. 5, DirectUNet | `prl-world-scl-rev.ipynb`, model class definition |
| Sec. 6, hyperparameter grid search (λ, α) | `prl-world-scl.ipynb`, grid-search cell (AUC-based selection) |
| Sec. 6, primary split / multi-seed / k-fold protocols | both notebooks, correspondingly labelled cells |
| Table 1 (discrete-operator diagnostics, K sweep) | reconstructed from the standalone flow analysis described in the paper; not part of the training notebooks (uses synthetic latent-shaped probes, not trained-encoder outputs — see manuscript Limitations) |
| Tables 2–5 (k-fold, variance decomposition, multi-seed, ablation) | parsed from `prl-world-scl.log` and `prl-world-scl-rev.log` as described above |

## Known limitations of this release

- The Table 1 discrete-flow diagnostics (decay curve, perturbation
  sensitivity) were produced with a standalone re-implementation of the SCL
  update on synthetic latent-shaped tensors, separate from the training
  notebooks, in order to isolate the operator's behaviour from the rest of
  the model. This script is not yet included in this release; it will be
  added before camera-ready if the paper is accepted. Until then, the
  update rule (Eq. 4 in the manuscript) is fully specified in the paper and
  in the `SCL Flow` class in `prl-world-scl.ipynb`, so it is reproducible
  from either.
- SCL-Input, SCL-Learnable, and the lesion-volume-tracking result are
  single-split only in both the paper and this code (not run under the
  full k-fold × multi-seed protocol); this is stated as a limitation in the
  manuscript.

## Citation

If you use this code, please cite the paper (full citation to be updated on
acceptance):

```bibtex
@article{savaliya2026worldcurved,
  title   = {World, Curved: Geometric Latent Priors for Longitudinal MS Forecasting},
  author  = {Savaliya, Maitri and Sinha, G. R.},
  journal = {Pattern Recognition Letters},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code: MIT License 
Note that the dataset itself is subject to its own Kaggle license and is not
included here.

## Contact

Maitri Savaliya — 22BT04062@gsfcuniversity.ac.in
Department of Computer Science and Engineering, GSFC University, Vadodara,
Gujarat, India
