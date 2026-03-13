<h2 align="center">
Transformer-Based Deep Learning for Protein Function Prediction:<br>
Experiments from the CAFA-6 Challenge
</h2>

<p align="center">
<strong>Vahe Sokhoyan</strong><br>
March 2026
</p>

## Overview

Predicting protein function from amino-acid sequence remains a central problem in computational biology. This study presents a systematic Machine Learning-based protein function prediction pipeline developed for the Critical Assessment of Functional Annotation 6 (CAFA-6) challenge hosted on Google Kaggle, achieving a top-30% leaderboard ranking as an individual contributor among more than 2000 competing teams and individuals. The pipeline addresses extreme class imbalance and highly variable label frequency across Gene Ontology (GO) annotations spanning three orders of magnitude.

Fixed-embedding approaches using extracted ProtT5-XL and ESM2 representations were systematically compared with end-to-end fine-tuned transformer architectures with partially or fully unfrozen internal layers, across multiple experimental settings. Additional aggregation and post-processing procedures were developed to account for the statistical structure of ensemble predictions beyond conventional mean or median aggregation across K-fold ensembles. At the post-processing stage, the hierarchical structure of the GO directed acyclic graph (GO-DAG) was exploited to recover biologically consistent functional annotations not directly identified by the ML models. Finally, a reference-guided denoising attempt using the UniProt/GOA database was performed and analyzed, revealing a systematic validation-leaderboard transfer gap with identified structural causes.

**Key aspects studied:**

- Fixed-embedding and end-to-end fine-tuned transformer architectures using ProtT5-XL (~3B parameters and ~1.2B parameters for the encode-only variant) and smaller ESM2 variants were systematically compared across multiple experimental settings, including large-scale GPU experimentation on NVIDIA A100 and AMD MI300X accelerators.
- An asymmetric neural architecture was designed leveraging the biological structure of the Gene Ontology: Molecular Function (MF) and Cellular Component (CC) representations inform Biological Process (BP) predictions without reverse information flow, improving BP performance while preserving MF and CC quality.
- The training dynamics was studied separately for labels spanning three orders of magnitude in frequency, revealing that the low-frequency tail continues to improve beyond the point where global validation metrics enter a regime of diminishing returns.
- Following identification of a strongly fold-sparse signal structure, an Information Accretion (IA)-aware ensemble aggregation strategy was developed, outperforming conventional mean and median aggregation by preserving rare-term signal that would otherwise be suppressed.
- GO-DAG propagation was applied as a biologically informed post-processing step, illustrated through GO-DAG-based plots of representative predicted terms.
- A reference-guided denoising strategy using UniProt/GOA evidence was developed and evaluated, demonstrating effective false-positive suppression on validation data while exposing the structural reasons for limited leaderboard transfer, including reference coverage mismatch, domain shift, and metric sensitivity effects.

---

**Keywords:** transformer fine-tuning, multi-label learning, extreme label imbalance, ensemble learning, ontology-aware machine learning, systematic experimentation, GPU-scale deep learning

---

# Data

## Data and Challenge Context

The Critical Assessment of Functional Annotation (CAFA-6) challenge evaluates computational methods for predicting protein function from amino-acid sequence. The dataset comprises approximately 82,000 training proteins with Gene Ontology (GO) annotations across three functional aspects: Molecular Function (MF), covering biochemical activities such as enzyme catalysis and binding; Biological Process (BP), covering biological pathways and systems-level functions; and Cellular Component (CC), covering subcellular localization. Performance is measured using the information accretion-weighted F-measure (Fmax), which accounts for both prediction accuracy and ontological specificity. Test evaluation is prospective, applied to proteins that were initially unannotated.

<p align="center">
<img src="figures/fig_cumulative_coverage_FPC_annotated.png" width="55%">
<img src="figures/fig_IA_vs_logpos_density_P_smoothed_white.png" width="44%">
</p>

<p align="center"><em><b>Figure 1.</b> Left: Cumulative annotation coverage versus retained terms (top-K). Right: Term specificity (IA) versus frequency.</em></p>

**Key data characteristics**

GO annotations follow a pronounced long-tail distribution (Fig. 1, left). A small subset of frequent terms accounts for the majority of annotations, while thousands of specific terms occur rarely. For the Biological Process ontology, approximately 2000 terms cover 80–85% of all annotations, yet the full ontology contains over 15,000 terms. Higher information accretion (IA) terms, encoding more specific biological functions, systematically occur at lower frequencies (Fig. 1, right), creating a direct coupling between term specificity and data sparsity.

This structure has concrete consequences for model design. Standard mean or median ensemble aggregation suppresses signal from rare but biologically relevant predictions. Global validation metrics are dominated by frequent terms and may obscure training dynamics in the rare-label tail. These observations directly motivated the asymmetric architecture design, IA-aware ensemble aggregation, and GO-DAG post-processing described in subsequent sections.

---

# Training Machine Learning models: Fixed embeddings vs. end-to-end transformer fine-tuning with unfrozen layers

One of the main aspects studied in this work is whether protein function prediction for the CAFA-6 task is best addressed using fixed transformer embeddings with downstream classifiers or by end-to-end fine-tuning of the transformer backbone. Both approaches were investigated systematically using the ProtT5-XL protein language model and, in a separate comparison, the ESM2 architecture.

In the fixed-embedding configuration, protein sequences are first encoded by the pretrained transformer model, and the resulting sequence representations are treated as frozen inputs to task-specific classification heads containing Multilayer perceptron (MLP) and attention layers. This approach is computationally efficient and widely used in bioinformatics applications, since the computationally heavy transformer inference step is performed only once and the downstream optimization is restricted to lightweight neural layers.

In the end-to-end configuration, the internal transformer representation is allowed to adapt to the downstream task by unfreezing selected transformer layers during training. In the ProtT5 configuration used here, the last two transformer blocks were unfrozen while the earlier blocks remained frozen.

---

<p align="center">
<img src="figures/cafa6_architecture_fixed.png" width="48%">
<img src="figures/cafa6_prott5_finetuning.png" width="50%">
</p>

<p align="center"><em>Figure 2. Model architecture used for protein function prediction. Left: multi-head prediction architecture operating in this example on fixed ProtT5 embeddings. Right: end-to-end fine-tuning scheme with the final transformer blocks of ProtT5 unfrozen and frozen earlier layers, where the sequence representation is processed by attention layers followed by ontology-specific MLP heads for MF, BP, and CC ontologies.</em></p>

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

<p align="center"><em>Table 2. Public leaderboard Fmax scores for different ensemble aggregation strategies applied to the five-fold Biological Process (P) predictions.</em></p>

---

# Summary

(unchanged content)
