<h2 align="center">
Transformer-Based Deep Learning for Protein Function Prediction:<br>
Experiments from the CAFA-6 Challenge
</h2>

<p align="center">
<strong>Vahe Sokhoyan</strong><br>
March 2026
</p>

# Overview

Predicting protein function from amino-acid sequence remains a central problem in computational biology. This study presents a systematic Machine Learning-based protein function prediction pipeline developed for the Critical Assessment of Functional Annotation 6 (CAFA-6) challenge hosted on Google Kaggle, achieving a top-30% leaderboard ranking as an individual contributor among more than 2000 competing teams and individuals. The pipeline addresses extreme class imbalance and highly variable label frequency across Gene Ontology (GO) annotations spanning three orders of magnitude.

Fixed-embedding approaches using extracted ProtT5-XL and ESM2 representations were systematically compared with end-to-end fine-tuned transformer architectures with partially or fully unfrozen internal layers, across multiple experimental settings. Additional aggregation and post-processing procedures were developed to account for the statistical structure of ensemble predictions beyond conventional mean or median aggregation across K-fold ensembles. At the post-processing stage, the hierarchical structure of the GO directed acyclic graph (GO-DAG) was exploited to recover biologically consistent functional annotations not directly identified by the ML models. Finally, a reference-guided denoising attempt using the UniProt/GOA database was performed and analyzed, revealing a systematic validation-leaderboard transfer gap with identified structural causes.

**Key aspects studied:**

- Fixed-embedding and end-to-end fine-tuned transformer architectures using ProtT5-XL (~3B parameters and ~1.2B parameters for the encode-only variant) and smaller ESM2 variants were systematically compared across multiple experimental settings, including large-scale GPU experimentation on NVIDIA A100 and AMD MI300X accelerators.
- An asymmetric neural architecture was designed leveraging the biological structure of the Gene Ontology: Molecular Function (MF) and Cellular Component (CC) representations inform Biological Process (BP) predictions without reverse information flow, improving BP performance while preserving MF and CC quality.
- The training dynamics was studied separately for labels spanning three orders of magnitude in frequency, revealing that the low-frequency tail continues to improve beyond the point where global validation metrics enter a regime of diminishing returns.
- Following identification of a strongly fold-sparse signal structure, an Information Accretion (IA)-aware ensemble aggregation strategy was developed, outperforming conventional mean and median aggregation by preserving rare-term signal that would otherwise be suppressed.
- GO-DAG propagation was applied as a biologically informed post-processing step, illustrated through GO-DAG-based plots of representative predicted terms.
- A reference-guided denoising strategy using UniProt/GOA evidence was developed and evaluated, demonstrating effective false-positive suppression on validation data while exposing the structural reasons for limited leaderboard transfer.

---

**Keywords:** transformer fine-tuning, multi-label learning, extreme label imbalance, ensemble learning, ontology-aware machine learning, systematic experimentation, GPU-scale deep learning

---

# Data

## Data and Challenge Context

The Critical Assessment of Functional Annotation (CAFA-6) challenge evaluates computational methods for predicting protein function from amino-acid sequence. The dataset comprises approximately 82,000 training proteins with Gene Ontology (GO) annotations across three functional aspects: Molecular Function (MF), Biological Process (BP), and Cellular Component (CC). Performance is measured using the information accretion-weighted F-measure (Fmax).

<p align="center">
<img src="figures/fig_cumulative_coverage_FPC_annotated.png" width="55%">
<img src="figures/fig_IA_vs_logpos_density_P_smoothed_white.png" width="44%">
</p>

<p align="center"><em><b>Figure 1.</b> Left: cumulative annotation coverage versus retained terms (top-K). Right: term specificity (IA) versus frequency.</em></p>

**Key data characteristics**

GO annotations follow a pronounced long-tail distribution. A small subset of frequent terms accounts for the majority of annotations, while thousands of specific terms occur rarely. Higher information accretion terms systematically occur at lower frequencies, creating a direct coupling between term specificity and data sparsity.

---

# Training Machine Learning models: Fixed embeddings vs. end-to-end transformer fine-tuning

<p align="center">
<img src="figures/cafa6_architecture_fixed.png" width="48%">
<img src="figures/cafa6_prott5_finetuning.png" width="50%">
</p>

<p align="center"><em>Figure 2. Model architecture used for protein function prediction. Left: multi-head prediction architecture operating on fixed ProtT5 embeddings. Right: end-to-end fine-tuning scheme with the final transformer blocks of ProtT5 unfrozen and frozen earlier layers, where the sequence representation is processed by attention layers followed by ontology-specific MLP heads for MF, BP, and CC ontologies.</em></p>

---

<p align="center">
<img src="figures/grid_FPC__fmax_ap__perfold_plus_mean_fixed.png" width="98%">
</p>

<p align="center"><em>Figure 3. Validation performance for the fixed-embedding ProtT5 baseline across five folds. Colored curves correspond to individual folds and the dashed line indicates the mean.</em></p>

---

<p align="center">
<img src="figures/grid_FPC__fmax_ap__perfold_plus_mean_tuned.png" width="98%">
</p>

<p align="center"><em>Figure 4. Validation curves for the end-to-end tuned ProtT5 model with two transformer layers unfrozen.</em></p>

---

## Training setup

(Table unchanged)

---

<p align="center">
<img src="figures/Fixed_vs_tuned_FMax_AP.png" width="98%">
</p>

<p align="center"><em>Figure 5. Comparison between fixed ProtT5 embeddings and end-to-end fine-tuning.</em></p>

---

# Effect of batch size

<p align="center">
<img src="figures/prott5_batch_compare_grid.png" width="98%">
</p>

<p align="center"><em>Figure 6. Influence of effective batch size on validation performance for the end-to-end ProtT5 configuration.</em></p>

---

# ProtT5 vs ESM2 comparison

<p align="center">
<img src="figures/Prott5_ESM2_nice_black.png" width="98%">
</p>

<p align="center"><em>Figure 7. Comparison of fine-tuning strategies for ProtT5 and ESM2.</em></p>

---

<p align="center">
<img src="figures/ESM2_full.png" width="98%">
</p>

<p align="center"><em>Figure 8. Fixed-embedding versus end-to-end tuning for the ESM2 model.</em></p>

---

# Rare-label dynamics

<p align="center">
<img src="figures/P_Fmax.png" width="33%">
<img src="figures/P_base_Fmax.png" width="33%">
<img src="figures/P_tail_Fmax.png" width="33%">
</p>

<p align="center"><em>Figure 9. Training dynamics of Fmax for the Biological Process ontology.</em></p>

---

<p align="center">
<img src="figures/P_AP.png" width="33%">
<img src="figures/P_base_AP.png" width="33%">
<img src="figures/P_tail_AP.png" width="33%">
</p>

<p align="center"><em>Figure 10. Training dynamics of Average Precision for the same subsets.</em></p>

---

# Aggregation and fold-support structure

<p align="center">
<img src="figures/P_fold_combo_top12.png" width="48%">
<img src="figures/P_single_fold_by_IAband.png" width="48%">
</p>

<p align="center"><em>Figure 11. Fold-support structure of predictions passing the base gate.</em></p>

---

<div align="center">

| Aggregation strategy | Leaderboard Fmax | Δ vs. single fold |
|---|:---:|:---:|
| Single fold (baseline) | 0.330 | — |
| Mean (3 folds) | 0.340 | +3.0% |
| Mean (5 folds) | 0.333 | +0.9% |
| Trimmed mean (5 folds) | 0.329 | −0.3% |
| Median (5 folds) | 0.324 | −1.8% |
| **IA pooling (5 folds)** | **0.348** | **+5.5%** |

</div>

<p align="center"><em>Table 2. Public leaderboard Fmax scores for different ensemble aggregation strategies.</em></p>

---

# GO-DAG propagation

<p align="center">
<img src="figures/GODAG_CC_nice.png" width="47%">
<img src="figures/GODAG_MF_nice.png" width="47%">
</p>

<p align="center"><em>Figure 12. GO-DAG neighborhood of representative predicted terms. The arrows point from more specific child terms toward more general parent terms.</em></p>

---

# Denoising diagnostics

<p align="center">
<img src="figures/Denoise1D.png" width="100%">
</p>

<p align="center"><em>Figure 13. Score distributions before and after UniProt/GOA-guided denoising.</em></p>

---

<p align="center">
<img src="figures/IA_vs_prob__FP__before__F__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="49%">
<img src="figures/IA_vs_prob__FP__after__F__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="49%">
</p>

<p align="center"><em>Figure 14. Term IA vs Meta-CatBoost probability before and after denoising.</em></p>

---

# Summary

This study presents a systematic investigation of transformer-based protein function prediction for the CAFA-6 challenge, covering architecture design, training dynamics, ensemble aggregation, ontology-aware post-processing, and reference-guided denoising.

Key findings include:

- **Transformer fine-tuning** improves performance over fixed embeddings.
- **Asymmetric architecture** improves BP predictions.
- **Rare-label tail continues improving** after global metrics plateau.
- **IA-aware pooling** provides the best leaderboard score.
- **GO-DAG propagation** improves biological consistency.
- **Reference-guided denoising** reveals systematic FP structure but does not transfer reliably to the leaderboard.

The computational experiments were conducted on NVIDIA A100 and AMD MI300X accelerators, and the author welcomes collaboration on extending these results toward peer-reviewed publication.
