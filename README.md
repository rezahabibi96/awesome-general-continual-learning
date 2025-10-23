# Awesome General Continual Learning (CL)

A curated and structured list of **Continual Learning (CL)** papers, focusing on taxonomy clarity and multi-label method categorization. A single paper may belong to multiple method families, and this list makes those relationships explicit and searchable.

---

## Table of Contents
- [CL in Vision](#cl-in-vision)
  - [Method Taxonomy](#method-taxonomy)
  - [Setting Taxonomy](#setting-taxonomy)
  - [Survey and Books](#survey-and-books)
  - [Index of Papers](#index-of-papers)
- [CL in Time Series](#cl-in-time-series)
  - [Setting Taxonomy](#setting-taxonomy-1)
  - [Index of Papers](#index-of-papers-1)
- [CL in Natural Language Processing (WIP)](#cl-in-natural-language-processing-wip)
- [CL in Reinforcement Learning (WIP)](#cl-in-reinforcement-learning-wip)
- [Acknowledgements](#acknowledgements)

---

## CL in Vision

### Method Taxonomy

- **Regularization-based**
  - Weight Regularization — `WR`
  - Knowledge Distillation
    - Feature-level — `KD-Feat`
    - Logit-level — `KD-Logit`
    - Relational-level — `KD-Rel`
    - Patch-level — `KD-Patch`
    - Prototype-level — `KD-Proto`

- **Replay-based**
  - Rehearsal Based
    - Data Space — `Rep-Data`
    - Feature Space — `Rep-Feat`
    - Label Space — `Rep-Label`
    - Embedding Space — `Rep-Embed`
  - Pseudo Replay
    - Generative Replay — `Gen-Data`
    - Feature Replay — `Gen-Feat`

- **Representation-based**
  - Self-supervised learning — `Rep-SSL`
  - Pre-training for Downstream Tasks
    - Fixed Backbone — `Rep-PT`
    - Updatable Backbone — `Rep-PT`
  - Adaptive Representation Learning — `Rep-ARL`
  - Template-based Classification
    - Prototype-based — `Rep-Proto`
    - Generative — `Rep-Gen`
    - Energy-based — `Rep-EBM`

- **Optimization-based**
  - Meta Learning — `Opt-Meta`
  - Gradient Projection — `Opt-GradProj`
  - Loss Landscape — `Opt-Loss`

- **Architecture-based**
  - Fixed-Capacity
    - Mask-based — `Arch-Mask`
    - Parameter Reallocation — `Arch-Realloc`
  - Capacity-increasing
    - Parameter Segregation — `Arch-Seg`
    - Model Decomposition — `Arch-Decomp`
    - Modular Network — `Arch-Mod`

### Setting Taxonomy

- **Task-Aware CL**
  - Task-IL — `TIL`
  - Class-IL — `CIL`
  - Domain-IL — `DIL`

- **General CL**
  - Online-CL — `OCL`
  - Task-Free CL — `TFCL`

- **Other CL Setting**
  - Continual Pre-training — `CPT`
  - Behaviour-IL / Environment-IL — `BEIL`
  - Few-Shot CL — `FSCL`

- **Other CL Application**
  - Object Detection — `OD`
  - Semantic Segmentation — `SS`
  - Conditional Generation — `CG`

### Survey and Books

### Index of Papers

| **Title** | **Year** | **Venue** | **CL Setting** | **CL Method** |
|-----------|:--------:|:---------:|:---------------:|------------------------|
| [Release the Potential of Memory Buffer in Continual Learning: A Dynamic System Perspective](https://ieeexplore.ieee.org/abstract/document/11198838) | 2025 | TPAMI | `OCL` | `KD-Rel`, `Rep-Data` |

---

## CL in Time Series

### Setting Taxonomy

- **Other CL Application**
  - Time Series Classification — `TSC`
  - Time Series Forecasting — `TSF`

### Index of Papers

| **Title** | **Year** | **Venue** | **CL Setting** | **CL Method** |
|-----------|:--------:|:---------:|:---------------:|------------------------|
| *(To be added)* | | | | |

---

## CL in Natural Language Processing (WIP)

### Method Taxonomy

### Setting Taxonomy

### Survey and Books

### Index of Papers

---

## CL in Reinforcement Learning (WIP)

### Method Taxonomy

### Setting Taxonomy

### Survey and Books

### Index of Papers

---

## Acknowledgements
Inspired by awesome lists in continual learning and lifelong learning research.
