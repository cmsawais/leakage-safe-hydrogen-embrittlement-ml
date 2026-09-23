# Data

The SEM micrographs are not included in this repository. They are available from the
corresponding author (muhammad.awais@phd.unipd.it) on request.

## Expected layout

```
data/
└── Mixed/
    ├── 316L_AR_01_001.png
    ├── 316L_AR_01_002.png
    ├── 316L_H2_02_001.png
    ├── X65_H2_03_001.png
    └── ...
```

## Filename convention

Every image is a `.png` whose stem has exactly four underscore-separated fields:

```
<material>_<condition>_<region>_<imageid>.png
```

| Field | Examples | Notes |
|-------|----------|-------|
| `material`  | `316L`, `X65`, `mX65`, `100Cr6`, `304` | alloy |
| `condition` | `AR`, `H2`, `AIR` | as-received / hydrogen-charged / air-exposed |
| `region`    | `01`, `02`, … | spatial region on the specimen — the **grouping unit** for Leave-One-Region-Out CV |
| `imageid`   | `001`, `002`, … | image index within the region |

Files that do not follow this pattern are silently skipped by `build_metadata()`.

The group identifier used for cross-validation is `<material>_<condition>_<region>`.
For the primary task in the paper (316L, AR vs H₂) this yields 14 groups: 8 AR regions
(18 images) and 6 H₂ regions (13 images). The full corpus of 143 images across all alloys
is used only for self-supervised autoencoder pretraining.
