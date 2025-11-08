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

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Progressive neural networks -- PNN | 2016 | ArXiv | `Arch-Mod` |  |
| Continual learning with deep generative replay -- DGR | 2017 | NeurIPS | `Gen-Data` | `TIL` |
| Expert gate: Lifelong learning with a network of experts -- EG | 2017 | CVPR | `KD-Rel`, `Arch-Mod` |  |
| Overcoming catastrophic forgetting in neural networks -- EWC | 2017 | PNAS | `WR` | `TIL` |
| FearNet: Brain-inspired model for incremental learning -- FearNet | 2017 | ArXiv | `Gen-Data` | `TIL` |
| iCaRL: Incremental classifier and representation learning -- iCaRL | 2017 | CVPR | `Rep-Data`, `Rep-Label`, `Repr-Proto` | `CIL` |
| Overcoming catastrophic forgetting by incremental moment matching -- IMM | 2017 | NeurIPS | `WR` | `TIL` |
| Gradient episodic memory for continual learning -- GEM | 2017 | NeurIPS | `Rep-Label`, `Opt-GradProj` | `TIL` |
| Learning without forgetting -- LwF | 2017 | TPAMI | `KD-Logit`, `Rep-Label` | `TIL` |
| PathNet: Evolution channels gradient descent in super neural networks -- PathNet | 2017 | ArXiv | `Arch-Mod` | `TIL` |
| Continual learning through synaptic intelligence -- SI | 2017 | PMLR | reg; | `TIL` |
| Variational continual learning -- VCL | 2017 | ArXiv | `WR` | `TIL` |
| On efficient lifelong learning with a-GEM -- A-GEM | 2018 | ArXiv | `Rep-Label`, `Opt-GradProj` | `TFCL` |
| Progress & compress: A scalable framework for continual learning -- P&C | 2018 | TMLR | `WR` | `TIL` |
| Lifelong learning with dynamically expandable networks -- DEN | 2018 | ICLR | `Arch-Mod` | `TIL` |
| Encoder based lifelong learning -- EBLL | 2018 | ICCV | `KD-Feat` |  |
| End-to-end incremental learning -- EEIL | 2018 | ECCV | `KD-Logit`, `Rep-Label` | `CIL` |
| Overcoming catastrophic forgetting with hard attention to the task -- HAT | 2018 | ICML | `Arch-Mask` | `TIL` |
| Memory aware synapses: Learning what (not) to forget -- MAS | 2018 | ECCV | `WR` | `TIL`, `OCL`, `TFCL` |
| Learning to learn without forgetting by maximizing transfer and minimizing interference -- MER | 2018 | ArXiv | `Opt-Meta` | `OCL` |
| Memory replay GANs: Learning to generate images from new categories without forgetting -- MeRGANs | 2018 | NeurIPS | `Gen-Data` | `TIL` |
| PackNet: Adding multiple tasks to a single network by iterative pruning -- Packnet | 2018 | CVPR | `Arch-Realloc` | `TIL` |
| Piggyback: Adapting a single network to multiple tasks by learning to mask weights -- Piggyback | 2018 | ECCV | `Arch-Mask` | `TIL` |
| Riemannian walk for incremental learning: Understanding forgetting and intransigence -- RWalk | 2018 | ECCV | `WR`, `Rep-Data` | `TIL`, `CIL` |
| Large scale incremental learning -- BiC | 2019 | CVPR | `Rep-Label` | `CIL` |
| Continual learning via neural pruning -- CLNP | 2019 | ArXiv | `Arch-Realloc` | `TIL` |
| Learning to remember: A synaptic plasticity driven framework for continual learning -- DGMa/DGMw/DGM | 2019 | CVPR | `Gen-Data`, `Arch-Seg` | `TIL` |
| Gradient based sample selection for online continual learning -- GSS | 2019 | NeurIPS | `Rep-Data` | `CIL`, `OCL`, `TFCL`, `FSCL` |
| Continual learning with hypernetworks -- HNET | 2019 | ArXiv | `WR`, `Gen-Data` | `TIL`, `CIL`, `TFCL` |
| Learning a unified classifier incrementally via rebalancing -- LUC | 2019 | CVPR | `WR`, `Rep-Feat` | `CIL` |
| Learning without memorizing -- LWM | 2019 | CVPR | `WR` | `CIL` |
| Il2M: Class incremental learning with dual memory -- IL2M | 2019 | ICCV | `Gen-Feat` | `CIL` |
| Online continual learning with maximal interfered retrieval -- MIR | 2019 | NeurIPS | `Rep-Data` | `OCL`, `TFCL` |
| Meta-learning representations for continual learning -- OML | 2019 | ICML | `Opt-Meta` | `OCL` |
| Continual learning of context-dependent processing in neural networks -- OWM | 2019 | Nature | `Opt-GradProj` | `CIL` |
| Prototype augmentation and self-supervised -- PASS | 2019 | CVPR | `Rep-Feat` |  |
| Rotate your networks: Better weight consolidation and less catastrophic forgetting -- R-EWC | 2019 | ICPR | `WR` |  |
| Random path selection for continual learning -- RPSNet | 2019 | NeurIPS | `Arch-Mod` |  |
| Uncertainty-based continual learning with adaptive regularization -- UCL | 2019 | NeurIPS | `Arch-Realloc` |  |
| Adversarial continual learning -- ACL | 2020 | ECCV | `Arch-Decomp` |  |
| Continual learning with node-importance based adaptive group sparse regularization -- AGS-CL | 2020 | NeurIPS | `Arch-Realloc` |  |
| Learning to continually learn -- ANML | 2020 | ECAI | `Opt-Meta` |  |
| Dark experience for general continual learning: A strong, simple baseline -- DER | 2020 | NeurIPS | `KD-Logit`, `KD-Proto`, `Rep-Data`, `Rep-Feat`, `Rep-Label`, `Repr-Proto` | `CIL` |
| Class-incremental learning via deep model consolidation -- DMC | 2020 | CVPR | `KD-Rel` |  |
| GAN memory with no forgetting -- GAN-memory | 2020 | NeurIPS | `Repr-Fix` |  |
| Gdumb: A simple approach that questions our progress in continual learning -- GDumb | 2020 | ECCV | `Rep-Data` | `TIL` |
| Generalized variational continual learning -- GVCL | 2020 | ArXiv | `Arch-Decomp` |  |
| A neural dirichlet process mixture model for task-free continual learning -- GRU-D | 2020 | ICLR | `Repr-Gen` |  |
| iTAML: An incremental task-agnostic meta-learning approach -- iTAML | 2020 | CVPR | `Opt-Meta` |  |
| Look-ahead meta learning for continual learning -- La-MAML | 2020 | NeurIPS | `Opt-Meta` |  |
| Representational continuity for unsupervised continual learning -- LUMP | 2020 | ArXiv | `Repr-SSL` |  |
| Merlin: Meta-consolidation for continual learning -- MERLIN | 2020 | NeurIPS | `Opt-Meta` | `OCL` |
| Mnemonics training: Multi-class incremental learning without forgetting -- Mnemonics | 2020 | CVPR | `Rep-Data`, `Rep-Feat`, `Gen-Data` |  |
| Orthogonal gradient descent for continual learning -- OGD | 2020 | AISTATS | `Opt-GradProj` |  |
| Continual learning in low-rank orthogonal subspaces -- OrthogSubspace | 2020 | NeurIPS | `Opt-GradProj` |  |
| Online fast adaptation and knowledge accumulation (OSAKA): A new approach to continual learning -- OSAKA | 2020 | NeurIPS | `Opt-Meta` |  |
| Remind your neural network to prevent catastrophic forgetting -- REMIND | 2020 | ECCV | `Gen-Feat` | `TIL` |
| Side-tuning: A baseline for network adaptation via additive side networks -- Side-Tuning | 2020 | ECCV | `Repr-Fix` |  |
| Understanding the role of training regimes in continual learning -- Stable-SGD | 2020 | NeurIPS | `Opt-Loss` |  |
| Supermasks in superposition -- SupSup | 2020 | NeurIPS | `Opt-GradProj` | `TIL`, `CIL` |
| Maintaining discrimination and fairness in class incremental learning -- WA | 2020 | CVPR | `Rep-Label` | `CIL` |
| Flattening sharpness for dynamic gradient projection memory benefits continual learning -- FS-DGPM | 2021 | NeurIPS | `Opt-GradProj` |  |
| Few-shot lifelong learning -- FSLL | 2021 | AAAI | `Repr-Proto` | `FSCL` |
| Gradient projection memory for continual learning -- GPM | 2021 | ArXiv | `Opt-GradProj` |  |
| Using hindsight to anchor past knowledge in continual learning -- HAL | 2021 | AAAI | `Rep-Data` |  |
| Continual learning via local module composition -- LMC | 2021 | NeurIPS | `Arch-Mod` |  |
| Optimizing reusable knowledge for continual learning via metalearning -- MARK | 2021 | NeurIPS | `Opt-Meta`, `Arch-Decomp` |  |
| Linear mode connectivity in multitask and continual learning -- MC-SGD | 2021 | ICLR | `Opt-Loss` |  |
| Efficient continual learning with modular networks and task-driven priors -- MNTDP | 2021 | ICLR | `Arch-Mod` |  |
| Natural continual learning: Success is a journey, not (just) a destination -- NCL | 2021 | NeurIPS | `Opt-GradProj` |  |
| Meta-learning with less forgetting on large-scale non-stationary task distributions (ORDER) -- ORDER | 2021 | ArXiv | `Repr-ARL` | `CPT` |
| Insights from the future for continual learning -- PODNet | 2021 | CVPR | `KD-Feat`, `Rep-Feat` | `TIL`, `CIL` |
| Posterior meta-replay for continual learning -- PR | 2021 | NeurIPS | `Opt-Meta` |  |
| Ss-il: Separated softmax for incremental learning -- SS-IL | 2021 | CVPR | `KD-Logit`, `Rep-Label` | `CIL` |
| Model zoo: A growing “brain” that learns continually -- Zoo | 2021 | NeurIPS | `Arch-Mod` |  |
| Balancing stability and plasticity through advanced null space in continual learning -- AdNS | 2022 | ArXiv | `Opt-GradProj` |  |
| Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation -- AFC | 2022 | CVPR | `KD-Feat` | `CIL` |
| Anti-retroactive interference for lifelong learning -- ARI | 2022 | ECCV | `Opt-Meta` |  |
| Self-supervised models are continual learners -- CaSSLe | 2022 | CVPR | `Repr-SSL` | `TIL`, `CIL`, `DIL` |
| CoSCL: Cooperation of small continual learners is stronger than a big one -- CoSCL | 2022 | ECCV | `Arch-Mod` |  |
| Mimicking the oracle: An initial phase decorrelation approach for class incremental learning -- CwD | 2022 | CVPR | `Repr-Fix` |  |
| DLCFT: Deep linear continual fine-tuning for general incremental learning -- DLCFT | 2022 | ArXiv | `Repr-Fix` |  |
| DualPrompt: Complementary prompting for rehearsal-free continual learning -- DualPrompt | 2022 | ECCV | `Repr-Fix` | `CIL`, `CPT` |
| DYTOX: Transformers for continual learning with DYnamic TOken EXpansion -- DYTOX | 2022 | CVPR | `Repr-Fix`, `Arch-Decomp` | `TIL` |
| Energy-based models for continual learning -- EBM-CL | 2022 | PMLR | `Repr-EBM` |  |
| Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima (f2m) -- F2M | 2022 | ArXiv | `Repr-Fix` | `FSCL` |
| Forward compatible few-shot class-incremental learning -- FACT | 2022 | CVPR | `Repr-Proto` | `FSCL` |
| Foster: Feature boosting and compression for class-incremental learning -- FOSTER | 2022 | CVPR | `Rep-Feat` | `CIL` |
| Helpful or harmful: Inter-task association in continual learning -- H2 | 2022 | ECCV | `Arch-Mask` |  |
| Incremental meta-learning via indirect discriminant alignment (IDA) -- IDA | 2022 | ArXiv | `Repr-ARL` | `CPT` |
| Generative negative text replay for continual vision-language pretraining (incCLIP) -- IncCLIP | 2022 | ArXiv | `Repr-ARL` | `CPT` |
| Learning to prompt for continual learning -- L2P | 2022 | CVPR | `Repr-Fix` | `TIL`, `CIL`, `DIL`, `CPT` |
| Towards better plasticity-stability trade-off in incremental learning: A simple linear connector -- Linear Connector | 2022 | CVPR | `Opt-Loss` |  |
| The challenges of continuous self-supervised learning -- MinRed | 2022 | ECCV | `Repr-SSL` |  |
| Continual learning with recursive gradient optimization -- RGO | 2022 | ICLR | `Opt-GradProj` |  |
| S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning -- S-Prompts | 2022 | NeurIPS | `Repr-Fix` | `TIL`, `DIL`, `CPT` |
| TRGP: Trust region gradient projection for continual learning -- TRGP | 2022 | ICLR | `Opt-GradProj` |  |
| Transfer without forgetting -- TwF | 2022 | ECCV | `Repr-Fix` |  |
| Forget-free continual learning with winning subnetworks -- WSN | 2022 | ICML | `Arch-Mask` |  |
| Class-incremental continual learning into the extended der-verse -- X-DER | 2022 | TPAMI | `Rep-Data` |  |
| Incorporating neuro-inspired adaptability for continual learning in artificial intelligence -- CAF | 2023 | Nature | `Arch-Mod` |  |
| Continual SLAM: Beyond lifelong simultaneous localization and mapping through continual learning -- CL-SLAM | 2023 | ArXiv | `Repr-SSL` |  |
| Continual momentum filtering on parameter space for online test-time adaptation -- CMF | 2023 | ICLR | `WR` | `OCL` |
| CODA-Prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning -- CODA-Prompt | 2023 | CVPR | `Repr-Fix` | `CIL`, `CPT` |
| Consistent prototype learning for few-shot continual relation extraction -- ConPL | 2023 | ACL | `Repr-Proto` | `FSCL` |
| EcoTTA: Memory-efficient continual test-time adaptation via self-distilled regularization -- EcoTTA | 2023 | CVPR | `KD-Feat` |  |
| Kalman filter online learning from non-stationary data -- KFOCL | 2023 | ICLR | `Repr-Upd` | `OCL`, `CPT` |
| PIVOT: Prompting for video continual learning -- PIVOT | 2023 | CVPR | `Repr-Fix` | `CIL` |
| Progressive prompts: Continual learning for language models -- Progressive-Prompts | 2023 | ICLR | `Repr-Fix` | `CPT` |
| An empirical investigation of the role of pre-training in lifelong learning -- SAM | 2023 | JMLR | `Repr-Upd`, `Opt-Meta`, `Opt-Loss` | `CPT` |
| SLCA: Slow learner with classifier alignment for continual learning on a pre-trained model -- SLCA | 2023 | ICCV | `Repr-Fix` |  |
| Few-shot class-incremental learning via training-free prototype calibration -- TEEN | 2023 | NeurIPS | `Repr-Proto` | `FSCL` |
| Decorate the newcomers: Visual domain prompt for continual test time adaptation -- VDP | 2023 | AAAI | `Repr-Fix` | `DIL` |
| Controlled low-rank adaptation with subspace regularization for continued training on large language models -- CLoRA | 2024 | ArXiv | `WR`, `Repr-ARL` |  |
| Elastic feature consolidation for cold start exemplar-free incremental learning -- EFC | 2024 | ICLR | `KD-Feat` | `CIL` |
| Fine-grained knowledge selection and restoration for non-exemplar class incremental learning -- FGKSR | 2024 | AAAI | `KD-Patch`, `KD-Proto` | `CIL`, `TFCL` |
| Locality sensitive sparse encoding for learning world models online -- Losse-FTL | 2024 | ArXiv | `Repr-ARL` | `OCL` |
| A probabilistic framework for modular continual learning -- PICLE | 2024 | ICLR | `Arch-Mod` |  |
| Adapter merging with centroid prototype mapping for scalable class-incremental learning -- ACMap | 2025 | CVPR | `Repr-Proto` |  |
| BiloRA: Almost-orthogonal parameter spaces for continual learning -- BiLoRA | 2025 | CVPR | `Repr-ARL` |  |
| C-loRA: Contextual low-rank adaptation for uncertainty estimation in large language models -- C-LoRA | 2025 | ArXiv | `Repr-ARL` |  |
| CL-LoRA: Continual low-rank adaptation for rehearsal-free class-incremental learning -- CL-LoRA | 2025 | CVPR | `Repr-ARL` | `CIL` |
| LoRA subtraction for drift-resistant space in exemplar-free continual learning -- DRS | 2025 | CVPR | `Repr-ARL` |  |
| Gated integration of low-rank adaptation for continual learning of language models -- GainLoRA | 2025 | ArXiv | `Repr-ARL` |  |
| Componential prompt-knowledge alignment for domain incremental learning -- KA-Prompt | 2025 | ArXiv | `Repr-ARL` | `DIL` |
| Prototype antithesis for biological few-shot class-incremental learning -- PA | 2025 | ICLR | `Repr-Proto` | `FSCL` |
| ProtoDepth: Unsupervised continual depth completion with prototypes -- ProtoDepth | 2025 | CVPR | `Repr-Proto` |  |
| RaSA: Rank-sharing low-rank adaptation -- RaSA | 2025 | ArXiv | `WR`, `Repr-ARL` |  |
| SD-LoRA: Scalable decoupled low-rank adaptation for class incremental learning -- SD-LoRA | 2025 | ArXiv | `Repr-ARL` |  |
| TreeloRA: Efficient continual learning via layer-wise loRAs guided by a hierarchical gradient-similarity tree -- TreeLoRA | 2025 | ArXiv | `Repr-ARL` |  |
| Unsupervised continual domain shift learning with multi-prototype modeling -- UCDSL/MPM | 2025 | CVPR | `Repr-Proto` | `DIL` |

---

## CL in Time Series

### Setting Taxonomy

- **Other CL Application**
  - Time Series Classification — `TSC`
  - Time Series Forecasting — `TSF`

### Index of Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|

---

## CL in Natural Language Processing (WIP)

### Survey and Books

### Index of Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Continual relation learning across domains -- EMAR | 2020 | ACL |  | `DIL` |
| Continual pre-training of language models for math problem understanding with syntax-aware memory network -- SNCL | 2022 | ACL |  | `CPT` |
| Continual pretraining of language models -- DAS | 2023 | ICLR |  | `CPT` |
| Lifelong language pretraining with distribution-specialized experts -- Lifelong-MoE | 2023 | ICML | `CPT` |  |
| Serial lifelong editing via mixture of knowledge experts -- ARM | 2025 | ACL |  | `TFCL` |
| HiDe-LLaVA: Hierarchical decoupling for continual instruction tuning of multimodal large language models -- hiDe-LLaVA | 2025 | ArXiv |  | `TFCIL` |
| Knowledge decoupling via orthogonal projection for lifelong editing of large language models -- KDE | 2025 | ACL |  | `TFCL` |
| Neuron-level sequential editing for large language models -- NSE | 2025 | ACL |  | `TFCL` |

---

## CL in Reinforcement Learning

### Survey and Books

### Index of Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Reinforced continual learning -- RCL | 2018 | NeurIPS |  | `TIL` |
| Online Continual Learning For Interactive Instruction Following Agents -- CAMA | 2024 | ArXiv |  | `OCl` `BEIL` |

---

## CL in Multimodal Domain

### Survey and Books

| **Title** | **Year** | **Venue** | **Type** | **CL Setting** |
|-----------|----------|-----------|----------|----------------|
| A practitioner’s guide to continual multimodal pretraining -- FoMo-in-Flux | 2024 | NeurIPS | benchmark paper | `CPT` |
| AVQACL: A novel benchmark for audio-visual question answering continual learning -- AVQACL | 2025 | CVPR | benchmark paper | `TIL` |

### Index of Papers

---

## Acknowledgements
Inspired by awesome lists in continual learning and lifelong learning research.
