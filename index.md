# GO-DAG Denoising for Protein Function Prediction (CAFA-6)

*Personal ML project — ongoing*

---

## Overview

This page documents a technical exploration of GO-DAG–aware denoising and fold-aware aggregation
in protein function prediction (CAFA-6).

The goal was to reduce false positives in low–mid probability regions while preserving recall,
using a combination of:

- transformer-based embeddings (ProtT5 / ESM variants)
- fold-aware IA pooling
- GO-DAG propagation
- targeted probability-window denoising

This is **not a formal paper** — it’s a reproducible project log and method summary.

---

## Key Result

![Main result](figures/fig1.png)

**Figure 1.** Example validation case showing separation of TP/FP after GO-DAG–aware denoising.

Public LB peak: ~0.36x (ensemble + denoising + IA pooling).

---

## Method (high level)

### Models

- Fine-tuned ProtT5 + ESM2 variants
- Character-level heads
- Siamese-style components for name / structure similarity

### Aggregation

- 5-fold CV
- IA-aware pooling across folds (not mean / median)
- selective GO-DAG propagation

### Denoising strategy

Applied only in FP-rich probability windows:

- suppress low-confidence descendants
- preserve ancestors
- aspect-specific thresholds (F / P / C)

Key idea:

> reduce noise where models are weakest, not where they are strongest.

---

## Diagnostics

![Style-1 plot](figures/fig_style1.png)

**Figure 2.** Style-1 plot: center-term confidence with ancestor structure.

![Style-2 plot](figures/fig_style2.png)

**Figure 3.** Style-2 plot: probability distribution vs true-term density.

These plots helped identify FP-heavy regions and tune denoising windows.

---

## Observations

- GO-DAG propagation alone increases recall but also FP.
- Surgical denoising in low bands improves precision without hurting recall.
- IA-aware fold pooling beats mean/median aggregation.
- UniProt-based veto helps in specific mid-confidence ranges but must be tightly gated.

---

## Code

Main experiments live here:

👉 https://github.com/YOURNAME/YOURREPO

Key components:

- embedding extraction
- fold aggregation
- GO-DAG propagation
- validation diagnostics
- submission writer

---

## Status

Active research project.

Next steps:

- RNA 3D Folding baseline
- cross-task transfer of denoising heuristics
- possible methods writeup later

---

## Contact

Vahe Sokhoyan  
(Data Scientist / ML Research)

