# Protein Function Prediction (CAFA-6)

Methods and diagnostics for ontology-constrained protein function prediction using large protein language models.

*Personal ML project — ongoing*

---

## Overview


- transformer-based embeddings (ProtT5 / ESM variants)
- fold-aware IA pooling
- GO-DAG propagation
- targeted probability-window denoising

This is **not a formal paper** — it’s a reproducible project log and method summary.

---

## Key Result

![Main result](figures/fig1.png)

**Figure 1.** Example validation case showing separation of TP/FP after GO-DAG–aware denoising.

## Method (high level)

### Models

### Aggregation

- 5-fold CV
- IA-aware pooling across folds (not mean / median)
- selective GO-DAG propagation

### Denoising strategy

## Diagnostics

![Style-1 plot](figures/fig_style1.png)

**Figure 2.** Style-1 plot: center-term confidence with ancestor structure.

![Style-2 plot](figures/fig_style2.png)

**Figure 3.** Style-2 plot: probability distribution vs true-term density.

These plots helped identify FP-heavy regions and tune denoising windows.

---

## Interpretation

## Suumary and Outlook

Vahe Sokhoyan  
(Data Scientist / ML Research)

