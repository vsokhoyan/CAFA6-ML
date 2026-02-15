# Protein Function Prediction (CAFA-6)

## Overview

This project presents an end-to-end protein function prediction pipeline developed for the CAFA-6 challenge on Google Kaggle, combining transformer-based embeddings, supervised machine-learning models, and ontology-aware post-processing. I compare fixed embeddings with end-to-end fine-tuned transformers, analyze performance across rare and frequent labels, and introduce GO-DAG–aware aggregation and denoising strategies to improve biological consistency of predictions. Beyond leaderboard metrics, I emphasize diagnostics and interpretability—showing how model outputs align with Gene Ontology structure and how specific predictions propagate to broader functional categories. The result is a practical, transparent workflow that balances predictive performance with biological plausibility, illustrated through quantitative validation and GO-DAG–based visual examples.

## Exploratory Data Analysis

![Main result](figures/fig1.png)

**Figure 1.** 

### Machine Learning Models

### Training Machine Learning models: Fixed embeddings vs. end-to-end transformer fine-tuning

<div style="display: flex; gap: 10px;">
  <img src="figures/cafa6_architecture5_fixed.png" width="48%">
  <img src="figures/cafa6_architecture5_fixed.png" width="48%">
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
