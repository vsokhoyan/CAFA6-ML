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

### Training Machine Learning models: Fixed embeddings vs. end-to-end transformer fine-tuning

<div style="display: flex; gap: 10px;">
  <img src="figures/cafa6_architecture_fixed.png" width="48%">
  <img src="figures/cafa6_prott5_finetuning.png" width="50%">
  <p align="center"><em>
Figure 2. Neural network architecture.
</em></p>
</div>

<div style="display: flex; gap: 10px;">
  <img src="figures/grid_FPC__fmax_ap__perfold_plus_mean_fixed.png" width="98%">
  <p align="center"><em>
Figure 2. Neural network architecture.
</em></p>
</div>

<div style="display: flex; gap: 10px;">
  <img src="figures/grid_FPC__fmax_ap__perfold_plus_mean_tuned.png" width="98%">
  <p align="center"><em>
Figure 2. Neural network architecture.
</em></p>
</div>

<div style="display: flex; gap: 10px;">
  <img src="figures/Fixed_vs_tuned_FMax_AP.png" width="98%">
  <p align="center"><em>
Figure 2. Mean validation Fmax (top row) and AP (bottom row) across 5 folds for fixed ProtT5 embeddings versus end-to-end fine-tuning of the last two transformer blocks. Fine-tuning consistently improves performance across all ontologies, with particularly strong gains for Biological Process (P). Tuned models converge faster and reach higher plateaus despite shorter training schedules, indicating meaningful adaptation of internal representations beyond downstream classifiers.
</em></p>
</div>

<div style="display: flex; gap: 10px;">
  <img src="figures/ESM2_full" width="98%">
  <p align="center"><em>
Figure 2. ESM2 fine-tuning.
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

Fold support and why aggregation matters (P aspect)

A five-fold model ensemble can be aggregated in multiple ways (mean, median, or more structured pooling). To understand how aggregation affects the retained signal, we analyzed—on the held-out test set—the fold support of protein–term events that pass a fixed “base gate” in at least one fold (same score window and IA minimum across all analyses). For each event, we recorded (i) the number of folds k (0–5) that pass the base gate, and (ii) the exact subset of folds that fired.

<div style="display: flex; gap: 10px;"> <img src="figures/P_fold_combo_top12.png" width="48%"> <img src="figures/P_k_support_counts.png" width="48%"> </div> <p style="margin-top: 6px; font-size: 0.95em;"> <b>Figure 1.</b> <i>Left:</i> Most frequent fold-combinations (top 12) for base-passing events in the P aspect; single-fold patterns {0}…{4} appear nearly as often as full five-fold agreement {0,1,2,3,4}, revealing substantial fold-sparse signal. <i>Right:</i> Histogram of k-support (k = number of folds passing the base gate) over the same candidate events; in P, a large fraction of passing events have k=1 and only a minority have k=5. </p>

For the Biological Process aspect (P), fold support is notably sparse. Figure 1 (left) shows that single-fold firing patterns {0}, {1}, {2}, {3}, {4} are among the most frequent outcomes—comparable in scale to full five-fold agreement {0,1,2,3,4}. Figure 1 (right) quantifies this sparsity via k-support: among events that pass in at least one fold, k=1 constitutes a large share, while k=5 constitutes a much smaller share. This immediately suggests that aggregation rules that implicitly require multi-fold consensus can suppress a substantial portion of P signal.

Mean vs median: “washout” is structural for P

Mean aggregation is often used to stabilize ensembles, but it has a specific failure mode when fold support is sparse: a strong score in one fold can be diluted by weaker scores in the remaining folds. We measured this effect explicitly by comparing any-fold passing vs mean-of-5 passing under the same base gate. Events that pass in at least one fold but fail after mean-of-5 are labeled LOST (“washout”). Because the mean cannot exceed the per-fold maximum, the opposite case (GAINED) is mathematically impossible when applying the same threshold.

<div style="display: flex; gap: 10px;"> 
  <img src="figures/P_washout_vs_mean5_by_IAband.png" width="48%"> 
  <img src="figures/P_k_support_by_IAband_counts.png" width="48%"> 
</div> <p style="margin-top: 6px; font-size: 0.95em;"> 
  <b>Figure 2.</b> <i>Left:</i> Washout vs mean-of-5 aggregation, stratified by IA band (LOW/MID/HIGH): “LOST” counts (any-fold passes, mean-of-5 fails) quantify how averaging can drop fold-sparse events; “GAINED” is zero by construction under a shared threshold. <i>Right:</i> k-support distributions stratified by IA band, showing how fold support varies across term information content. </p>

Figure 2 (left) shows that washout is substantial in P: a large number of events that pass in at least one fold fall below the base gate after averaging. Stratifying by information content (IA) indicates that washout is concentrated in the low–mid IA regime, i.e., it is not restricted to only the most extreme rare-term tail. This provides a mechanistic explanation for why simple averaging can be detrimental in P: a large portion of events are supported by only 1–2 folds, and averaging pushes many of these below the gate.

Median aggregation is even more restrictive in this setting. With five folds, the median behaves like a consensus filter: if fewer than ~3 folds produce a high score, the median remains low. Since P contains a large mass of k=1–2 events (Figure 2, right), median aggregation disproportionately suppresses them compared to the mean.

IA-aware pooling: preserving rare/specific signal without destabilizing common terms

To reduce washout while keeping common terms stable, we used IA-aware pooling. The key idea is to let the aggregation rule depend on term information content (IA), which correlates with rarity/specificity:

Low IA (common terms): use the mean over all 5 folds (stable consensus).

Mid IA: use the mean over the top-3 folds (reduces dilution of partial support).

High IA (rare/specific terms): use the mean over the top-2 folds (preserves strong fold-local evidence).

<div style="display: flex; gap: 10px;"> 
  <img src="figures/P_k_support_by_IAband_norm.png" width="48%"> 
  <img src="figures/P_single_fold_by_IAband.png" width="48%"> 
</div> <p style="margin-top: 6px; font-size: 0.95em;"> 
  <b>Figure 3.</b> <i>Left:</i> k-support distributions by IA band (normalized within each band over k=1..5), highlighting that fold support structure depends on term information content. <i>Right:</i> Single-fold firing (k=1) by IA band and fold index, showing that sparse-support events are common and not dominated by one particular fold. </p>

Figure 3 links the aggregation choice directly to the empirical structure of the ensemble. The IA-stratified k-support (left) shows that fold support varies with term IA, while the single-fold analysis (right) demonstrates that sparse-support events are common and distributed across folds rather than being driven by a single outlier fold. IA-aware pooling exploits this structure: it retains robustness for low-IA, high-consensus terms while mitigating the systematic dilution that mean (and especially median) impose on higher-IA predictions.

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

### Denoising strategy and empirically observed limitations

Late in the competition, I experimented with a reference-guided denoising step that uses UniProt/GOA evidence as a conservative “sanity filter” for low-to-mid confidence predictions. The motivation was empirical: in validation, a large fraction of false positives (FPs) clustered in a characteristic score regime and formed IA-banded “stripes” in the 2D spectrum (term IA vs. predicted probability). Many of these events looked like plausible model artifacts rather than biology.

Method (UniProt/GOA-guided downscaling / veto).
After aggregation and GO-DAG propagation, I ran a lightweight Meta-CatBoost classifier on the Top-M candidates per protein (here: M=250 for F and M=300 for C) to produce a calibrated Meta-CatBoost probability. In a configurable probability band (here 0.20–0.60 for F/C), I then checked whether the predicted GO term is supported by UniProt/GOA for that protein. If not supported, the score was deterministically downscaled (and optionally hard-vetoed if the downscaled score fell below the band edge). Importantly, high-confidence predictions were not touched (a separate skip threshold prevents denoising above a high score).

This produces a clear qualitative signature on validation: FP mass is removed “surgically” in the affected score region, while the TP structure is much less affected.

<div style="display: flex; gap: 10px;"> <img src="figures/ProbDist__TP_FP__before_after__F__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="48%"> <img src="figures/ProbDist__TP_FP__before_after__C__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="48%"> </div> <p align="center"><em> Figure D1. Score distributions of retained Top-M candidates (Meta-CatBoost probability &gt; 0.20), showing True Positives (TP) and False Positives (FP) before vs. after UniProt/GOA-guided denoising. Left: Molecular Function (F). Right: Cellular Component (C). In both aspects, denoising strongly suppresses FP density in the low-to-mid score region while leaving the TP distribution comparatively stable. </em></p> <div style="display: flex; gap: 10px;"> <img src="figures/IA_vs_prob__FP__before__F__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="48%"> <img src="figures/IA_vs_prob__FP__after__F__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="48%"> </div> <p align="center"><em> Figure D2. 2D spectrum for False Positives (F): term Information Accretion (IA) vs Meta-CatBoost probability (probability &gt; 0.20). Left: before denoising. Right: after UniProt/GOA-guided denoising. The denoising step removes a substantial portion of FP mass in the targeted probability regime, visibly “carving out” structured IA-banded stripe patterns that are prominent before denoising. </em></p> <div style="display: flex; gap: 10px;"> <img src="figures/IA_vs_prob__FP__before__C__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="48%"> <img src="figures/IA_vs_prob__FP__after__C__GODAG__den1__band0.20-0.60__minp0.20__rows20000.png" width="48%"> </div> <p align="center"><em> Figure D3. 2D spectrum for False Positives (C): term IA vs Meta-CatBoost probability (probability &gt; 0.20), before vs. after denoising. As in F, the denoising step selectively suppresses FP-dense structures in the low-to-mid probability range, consistent with a reference-guided removal of spurious predictions rather than a global score shift. </em></p>

Empirically observed limitation: strong validation effect, weak leaderboard transfer.
Despite being highly effective on validation diagnostics (precision increased markedly in the targeted score range, with minimal apparent TP loss), the same denoising did not reliably improve the public leaderboard (often neutral or slightly detrimental). This mismatch is a useful cautionary example of how “clean” validation wins can fail to transfer:

Reference coverage mismatch: UniProt/GOA support is incomplete and uneven across proteins, taxa, and term granularity. A term missing from GOA for a protein is not necessarily false—so a reference-guided downscale can silently remove true-but-unrecorded functions, especially in harder test regions.

Domain shift in the hidden test set: the competition’s hidden labels may differ from train-derived validation in annotation density and completeness. A method tuned to suppress validation FPs can become over-conservative when ground truth is sparse or differently curated.

Metric sensitivity and thresholding effects: CAFA-style metrics are sensitive to score calibration and global threshold behavior. A local improvement in “FP cleanliness” within a band does not guarantee a better global operating point after threshold selection, and small probability shifts can move the optimum threshold unfavorably.

Interaction with GO-DAG propagation: denoising happens after ontology-aware propagation in this pipeline. Removing (or downscaling) a child prediction can indirectly reduce useful propagated parent signal, which may matter more on the test distribution than on the validation slice.

Overall, this denoising strategy was still valuable: it exposed a diagnostic handle on systematic FP structure (the IA–probability “stripes”) and provided a concrete example of how to build controlled, reference-aware post-processing. At the same time, its limited leaderboard transfer highlights a general lesson from CAFA-6: strong validation diagnostics are necessary but not sufficient—robust gains require methods that tolerate reference incompleteness and distribution shifts without becoming overly conservative.

## Summary and Outlook

## Bibliography
