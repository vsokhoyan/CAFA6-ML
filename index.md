# Protein Function Prediction (CAFA-6)

## Overview

## Exploratory Data Analysis

![Main result](figures/fig1.png)

**Figure 1.** 

### Machine Learning Models

### Training models on fixed embeddings vs. end-to-end transformer fine-tuning

![Main result](figures/Diagram_training_v0.png)

### Training on rare labels vs. main training body

<div style="display: flex; gap: 10px;">
  <img src="figures/P_Fmax.png" width="33%">
  <img src="figures/P_base_Fmax.png" width="33%">
  <img src="figures/P_tail_Fmax.png" width="33%">
</div>

<div style="display: flex; gap: 10px;">
  <img src="figures/P_AP.png" width="33%">
  <img src="figures/P_base_AP.png" width="33%">
  <img src="figures/P_tail_AP.png" width="33%">

  <p align="center"><em>
Figure 2. Diagnostic comparison across three validation cases.
</em></p>
</div>

### Aggregation

### GO-DAG propagation

<div style="display: flex; gap: 10px;">
  <img src="figures/S2_prop_demo_val_C_row27289_GO_0005664.png" width="47%">
  <img src="figures/S2_prop_demo_val_C_row27289_GO_0005664.png" width="47%">
</div>

### Denoising strategy

## Diagnostics

![Style-1 plot](figures/fig_style1.png)

**Figure 2.** Style-1 plot: center-term confidence with ancestor structure.

![Style-2 plot](figures/fig_style2.png)

**Figure 3.** Style-2 plot: probability distribution vs true-term density.

## Interpretation

## Summary and Outlook
