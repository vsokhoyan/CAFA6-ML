# Protein Function Prediction (CAFA-6)

## Overview

This project presents an end-to-end protein function prediction pipeline developed for the CAFA-6 challenge on Google Kaggle, combining transformer-based embeddings, supervised machine-learning models, and ontology-aware post-processing. I compare fixed embeddings with end-to-end fine-tuned transformers, analyze performance across rare and frequent labels, and introduce GO-DAG–aware aggregation and denoising strategies to improve biological consistency of predictions. Beyond leaderboard metrics, I emphasize diagnostics and interpretability—showing how model outputs align with Gene Ontology structure and how specific predictions propagate to broader functional categories. The result is a practical, transparent workflow that balances predictive performance with biological plausibility, illustrated through quantitative validation and GO-DAG–based visual examples.

## Exploratory Data Analysis

## Exploratory Data Analysis

<div style="display: flex; gap: 10px;">
  <img src="figures/fig_label_frequency_spectra_FPC.png" width="48%">
  <img src="figures/fig_topK_cum_coverage_P.png" width="48%">
</div>

<p align="center"><em>
Figure E1 (left). Train label frequency spectra for Molecular Function (F), Biological Process (P), and Cellular Component (C). The number of GO terms is plotted against the number of annotated proteins per term (log–log). All aspects exhibit a pronounced long-tail distribution: a small number of frequent terms and a very large number of rare ones.
<br><br>
Figure E2 (right). Cumulative positive coverage for Biological Process (P) as a function of K, where only the K most frequent labels are retained. The curve shows a clear knee: coverage rises rapidly for small K and then saturates more slowly, providing a quantitative basis for selecting a head vocabulary size (e.g., K≈2000–3000) before treating the long tail separately.
</em></p>

<div style="display: flex; gap: 10px;">
  <img src="figures/fig_topK_frequency_threshold_P.png" width="48%">
  <img src="figures/fig_IA_vs_label_frequency_density_P_log.png" width="48%">
</div>

<p align="center"><em>
Figure E3 (left). Frequency threshold vs rank for P terms: the number of positive proteins associated with the K-th most frequent label (log–log). This illustrates how rapidly terms become extremely sparse beyond a chosen K, helping assess whether additional labels are learnable from the available supervision.
<br><br>
Figure E4 (right). Density map of Information Accretion (IA) versus label frequency for P (log-intensity). Higher-IA terms tend to be rarer, linking ontology specificity to data sparsity and motivating IA-aware calibration and post-processing strategies later in the pipeline.
</em></p>

<div style="display: flex; gap: 10px;">
  <img src="figures/fig_train_sequence_lengths_loglog.png" width="48%">
  <img src="figures/fig_labels_per_protein_FPC_total.png" width="48%">
</div>

<p align="center"><em>
Figure E5 (left). Train sequence length distribution (log–log). Protein lengths span several orders of magnitude, motivating careful batching, truncation, and compute-aware design when fine-tuning transformer models.
<br><br>
Figure E6 (right). Distribution of annotation counts per protein for F/P/C and total (log scale). Most proteins carry relatively few labels, while a smaller subset is densely annotated, contributing to large variance across proteins and folds.
</em></p>

### Summary of EDA insights

Before training, I inspected label and sequence statistics to understand class imbalance, annotation sparsity, and whether a restricted label vocabulary would be justified. This avoids “blind” modeling and directly informs later architectural and post-processing choices.

The label-frequency spectra reveal an extreme long-tail structure across all GO aspects, implying that naive multi-label training is dominated by negatives and that performance differs strongly between frequent “head” terms and rare “tail” terms. The top-K coverage and threshold plots show that, especially for Biological Process, a relatively small subset of labels already explains a large fraction of positive annotations, while labels beyond K≈2000–3000 become extremely sparse. This motivates training on a controlled head vocabulary and treating the tail separately.

The IA–frequency density further shows that high-information (more specific) terms are typically rare, providing a principled link between ontology specificity and data sparsity. This observation later motivates IA-aware calibration and GO-DAG–based aggregation rules.

Finally, the wide distribution of protein lengths and the sparse per-protein annotation counts highlight practical constraints for transformer fine-tuning (memory, truncation) and explain why fold-level variance and tail performance require special attention.

Together, these diagnostics establish the statistical structure of the CAFA-6 dataset and motivate the modeling, aggregation, and denoising strategies described in the following sections.


### Machine Learning Models

### Training Machine Learning models: Fixed embeddings vs. end-to-end transformer fine-tuning

<div style="display: flex; gap: 10px;">
  <img src="figures/cafa6_architecture_fixed.png" width="48%">
  <img src="figures/cafa6_prott5_finetuning.png" width="50%">
  <p align="center"><em>
Figure 2. Neural network architecture.
</em></p>

</div>


### Training on rare labels vs. main training body

<div style="display: flex; gap: 10px;">
  <img src="figures/P_Fmax.png" width="33%">
  <img src="figures/P_base_Fmax.png" width="33%">
  <img src="figures/P_tail_Fmax.png" width="33%">
  <p align="center"><em>
Figure 2. Diagnostic comparison across three validation cases for Fmax.
</em></p>

</div>

<div style="display: flex; gap: 10px;">
  <img src="figures/P_AP.png" width="33%">
  <img src="figures/P_base_AP.png" width="33%">
  <img src="figures/P_tail_AP.png" width="33%">

  <p align="center"><em>
Figure 2. Diagnostic comparison across three validation cases for AP. 
</em></p>
</div>

### Aggregation



### GO-DAG propagation
GO terms are not independent labels: they live in a directed acyclic graph (DAG) where edges encode “is-a” / “part-of” relationships. In practice, if a protein is predicted with high confidence for a specific term, it is often reasonable—and biologically consistent—to assign some confidence to its ancestors (more general terms). I use this structure as a lightweight post-processing step: after aggregating model outputs (e.g., across folds), I propagate signal upward (child → parent) under controlled rules (depth, minimum score threshold, and propagation strength), which can improve ontology-consistency and sometimes recall while keeping precision under control.
Figure X provides a qualitative illustration: predicted center terms sit on short, meaningful ontology paths, showing how the GO-DAG encodes “specific implies general” relationships that we can leverage during post-aggregation.

<div style="display: flex; gap: 10px;">
  <img src="figures/GODAG_plot1.png" width="47%">
  <img src="figures/GODAG_plot2.png" width="47%">
    
  <p align="center"><em>
Figure 2. Left (C): A predicted cellular-component term (center) is shown together with nearby GO ancestors/descendants in the GO directed acyclic graph. Arrows point upward in the ontology (from more specific child terms toward more general parent terms), illustrating how related subcellular-location terms cluster along a small number of biologically meaningful paths.
Right (F): A predicted molecular-function term (center) is connected to increasingly general DNA-binding terms. The directed edges visualize how a specific binding activity implies membership in broader functional categories, providing a mechanistic sanity-check for model predictions via ontology structure.
</em></p>
</div>

### Denoising strategy

## Diagnostics

![Style-1 plot](figures/fig_style1.png)

**Figure 2.** Style-1 plot: center-term confidence with ancestor structure.

![Style-2 plot](figures/fig_style2.png)

**Figure 3.** Style-2 plot: probability distribution vs true-term density.

## Interpretation

## Summary and Outlook
