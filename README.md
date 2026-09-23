# Leakage-Safe Machine Learning for Hydrogen Embrittlement Detection in 316L Stainless Steel

Code accompanying the manuscript

> **Leakage-Safe Machine Learning for Hydrogen Embrittlement Detection in 316L Stainless Steel: A Region-Held-Out Evaluation of Texture and Deep Features in SEM Micrographs**
> M. Awais, M. Yaseen, A. Shakoor, N. A. Niaz, H. Zia, M. Z. Shakoor

The repository contains the full analysis pipeline used to produce the results in the paper:
Leave-One-Region-Out (LORO) cross-validation of six feature–classifier combinations on
SEM micrographs of as-received (AR) and hydrogen-charged (H₂) 316L, a group-level
permutation test, and Grad-CAM interpretability.

## Contents

```
.
├── notebooks/
│   └── sem_enhanced_pipeline.ipynb   # main pipeline (Table 1, Table 2, Figs. 2–5, LBP+GLCM+SVM permutation test)
├── scripts/
│   └── rerun_permutation_lbp_svm.py  # group-level permutation test for the best model, LBP+SVM (Fig. 4)
├── data/                             # SEM images go here (see data/README.md)
├── figures/                          # output figures are written here
├── results/                          # output tables / null distributions are written here
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

## Pipeline overview

| Step | Where | What it does |
|------|-------|--------------|
| 0 | notebook, "Load Dataset" | Parses `material_condition_region_imageid.png` filenames into a metadata table (143 images, 4 alloys); selects the 316L AR vs H₂ task (31 images, 14 regions: 8 AR, 6 H₂). |
| 1 | notebook, "Plan 1" | Trains a convolutional autoencoder for 30 epochs on all 143 unlabelled images (MSE reconstruction loss); the frozen encoder yields 64-d global embeddings (`SSL_global`). |
| 2 | notebook, "Plan 2" | Extracts classical texture descriptors: uniform LBP (P = 24, r = 3; 26-d) and GLCM (d = 1, 3, 5 px; 4 angles; 6 Haralick properties; 72-d). |
| 3 | notebook, "Plan 3" | 14-fold Leave-One-Region-Out CV for `SSL_global`, `LBP+SVM`, `GLCM+SVM`, `LBP+GLCM+SVM`, `LBP+GLCM+RF`, and a compact CNN trained from scratch per fold. Pooled out-of-fold predictions give Table 1 and the confusion matrix (Table 2 / Fig. 1); per-fold balanced accuracies give Figs. 2 and 3. |
| 4 | notebook, "Plan 4" and `scripts/rerun_permutation_lbp_svm.py` | Group-level permutation test: region-to-label assignments are permuted (500 draws from the C(14,6) = 3003 arrangements) and the whole LORO pipeline is re-run for each. The notebook runs it for `LBP+GLCM+SVM` (p = 0.174); the script runs the identical procedure for the best model, `LBP+SVM` (p = 0.008, Fig. 4). |
| 5 | notebook, "Plan 5" | Trains the CNN on all 316L images (20 epochs, no held-out split) and produces Grad-CAM overlays for three AR and three H₂ images (Fig. 5). |

All experiments use `seed = 42` and run on CPU.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The results in the paper were produced with Python 3.13, PyTorch 2.7.0 and scikit-learn 1.6 on CPU
(`CUDA_VISIBLE_DEVICES` is set to an empty string in the code so that CPU execution is enforced).

## Data

The SEM image set is not distributed with this repository. It is available from the
corresponding author on request. Place the images in `data/Mixed/` following the naming
convention described in `data/README.md`.

## Running

**Main pipeline (Tables 1–2, Figures 2–5):**

```bash
jupyter lab notebooks/sem_enhanced_pipeline.ipynb
```

Run all cells top to bottom. The per-fold LORO loop trains a small CNN in every fold, so a full
run takes on the order of a few minutes on a modern CPU.

**Permutation test for LBP+SVM (Figure 4):**

```bash
python scripts/rerun_permutation_lbp_svm.py
# or, if the data live elsewhere:
SEM_DATA_DIR=/path/to/Mixed python scripts/rerun_permutation_lbp_svm.py
```

This writes `figures/permutation_test_lbp_svm.png` and `results/permutation_null_lbp_svm.csv`.

## Expected results

Pooled LORO results for 316L AR vs H₂ (14 folds, 31 images), as reported in Table 1:

| Model | Balanced Acc. | F1 (H₂) | Precision (H₂) | Recall (H₂) |
|-------|--------------:|--------:|---------------:|------------:|
| LBP+SVM      | 0.79 | 0.75 | 0.82 | 0.69 |
| CNN          | 0.67 | 0.67 | 0.55 | 0.85 |
| LBP+GLCM+SVM | 0.62 | 0.52 | 0.60 | 0.46 |
| LBP+GLCM+RF  | 0.59 | 0.50 | 0.55 | 0.46 |
| SSL_global   | 0.56 | 0.48 | 0.50 | 0.46 |
| GLCM+SVM     | 0.52 | 0.36 | 0.44 | 0.31 |

Permutation test (500 region-level permutations): LBP+SVM observed BA = 0.7906, p = 0.008;
LBP+GLCM+SVM observed BA = 0.6197, p = 0.174.

Because the CNN folds involve stochastic data augmentation and weighted sampling, the CNN row
may differ slightly across platforms and library versions; the classical-descriptor rows and the
permutation test are deterministic given the seed.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`).

## License

MIT — see `LICENSE`.
