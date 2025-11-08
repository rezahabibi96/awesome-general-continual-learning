# Awesome General Continual Learning (CL)

A curated and structured list of **Continual Learning** papers

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

### Survey Papers

| **Title** | **Year** | **Venue** | **Type** | **Setting** |
|-----------|----------|-----------|----------|-------------|
| Continual learning: A systematic literature review | 2025 | Neural Networks | survey paper |  |
| A reality check on pre-training for exemplar-free class-incremental learning | 2025 | CVPR | benchmark paper |  |
| Class-incremental learning: A survey | 2024 | TPAMI | survey paper |  |
| A comprehensive survey of continual learning: theory, method and application | 2024 | TPAMI | survey paper |  |
| Continual learning with pre-trained models: a survey | 2024 | IJCAI | survey paper |  |
| Recent advances of continual learning in computer vision: An overview | 2024 | ArXiv | survey paper |  |
| Catastrophic forgetting in deep learning: A comprehensive taxonomy | 2024 | ArXiv | survey paper |  |

### Research Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Unsupervised continual domain shift learning with multi-prototype modeling -- UCDSL/MPM | 2025 | CVPR | `Repr-Proto` | `DIL` |
| TreeloRA: Efficient continual learning via layer-wise loRAs guided by a hierarchical gradient-similarity tree -- TreeLoRA | 2025 | ArXiv | `Repr-ARL` |  |
| SD-LoRA: Scalable decoupled low-rank adaptation for class incremental learning -- SD-LoRA | 2025 | ArXiv | `Repr-ARL` |  |
| RaSA: Rank-sharing low-rank adaptation -- RaSA | 2025 | ArXiv | `WR`, `Repr-ARL` |  |
| ProtoDepth: Unsupervised continual depth completion with prototypes -- ProtoDepth | 2025 | CVPR | `Repr-Proto` |  |
| Prototype antithesis for biological few-shot class-incremental learning -- PA | 2025 | ICLR | `Repr-Proto` | `FSCL` |
| Componential prompt-knowledge alignment for domain incremental learning -- KA-Prompt | 2025 | ArXiv | `Repr-ARL` | `DIL` |
| Gated integration of low-rank adaptation for continual learning of language models -- GainLoRA | 2025 | ArXiv | `Repr-ARL` |  |
| LoRA subtraction for drift-resistant space in exemplar-free continual learning -- DRS | 2025 | CVPR | `Repr-ARL` |  |
| CL-LoRA: Continual low-rank adaptation for rehearsal-free class-incremental learning -- CL-LoRA | 2025 | CVPR | `Repr-ARL` | `CIL` |
| C-loRA: Contextual low-rank adaptation for uncertainty estimation in large language models -- C-LoRA | 2025 | ArXiv | `Repr-ARL` |  |
| BiloRA: Almost-orthogonal parameter spaces for continual learning -- BiLoRA | 2025 | CVPR | `Repr-ARL` |  |
| Adapter merging with centroid prototype mapping for scalable class-incremental learning -- ACMap | 2025 | CVPR | `Repr-Proto` |  |
| A probabilistic framework for modular continual learning -- PICLE | 2024 | ICLR | `Arch-Mod` |  |
| Locality sensitive sparse encoding for learning world models online -- Losse-FTL | 2024 | ArXiv | `Repr-ARL` | `OCL` |
| Fine-grained knowledge selection and restoration for non-exemplar class incremental learning -- FGKSR | 2024 | AAAI | `KD-Patch`, `KD-Proto` | `CIL`, `TFCL` |
| Elastic feature consolidation for cold start exemplar-free incremental learning -- EFC | 2024 | ICLR | `KD-Feat` | `CIL` |
| Controlled low-rank adaptation with subspace regularization for continued training on large language models -- CLoRA | 2024 | ArXiv | `WR`, `Repr-ARL` |  |
| Decorate the newcomers: Visual domain prompt for continual test time adaptation -- VDP | 2023 | AAAI | `Repr-Fix` | `DIL` |
| Few-shot class-incremental learning via training-free prototype calibration -- TEEN | 2023 | NeurIPS | `Repr-Proto` | `FSCL` |
| SLCA: Slow learner with classifier alignment for continual learning on a pre-trained model -- SLCA | 2023 | ICCV | `Repr-Fix` |  |
| An empirical investigation of the role of pre-training in lifelong learning -- SAM | 2023 | JMLR | `Repr-Upd`, `Opt-Meta`, `Opt-Loss` | `CPT` |
| Progressive prompts: Continual learning for language models -- Progressive-Prompts | 2023 | ICLR | `Repr-Fix` | `CPT` |
| PIVOT: Prompting for video continual learning -- PIVOT | 2023 | CVPR | `Repr-Fix` | `CIL` |
| Kalman filter online learning from non-stationary data -- KFOCL | 2023 | ICLR | `Repr-Upd` | `OCL`, `CPT` |
| EcoTTA: Memory-efficient continual test-time adaptation via self-distilled regularization -- EcoTTA | 2023 | CVPR | `KD-Feat` |  |
| Consistent prototype learning for few-shot continual relation extraction -- ConPL | 2023 | ACL | `Repr-Proto` | `FSCL` |
| CODA-Prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning -- CODA-Prompt | 2023 | CVPR | `Repr-Fix` | `CIL`, `CPT` |
| Continual momentum filtering on parameter space for online test-time adaptation -- CMF | 2023 | ICLR | `WR` | `OCL` |
| Continual SLAM: Beyond lifelong simultaneous localization and mapping through continual learning -- CL-SLAM | 2023 | ArXiv | `Repr-SSL` |  |
| Incorporating neuro-inspired adaptability for continual learning in artificial intelligence -- CAF | 2023 | Nature | `Arch-Mod` |  |
| Class-incremental continual learning into the extended der-verse -- X-DER | 2022 | TPAMI | `Rep-Data` |  |
| Forget-free continual learning with winning subnetworks -- WSN | 2022 | ICML | `Arch-Mask` |  |
| Transfer without forgetting -- TwF | 2022 | ECCV | `Repr-Fix` |  |
| TRGP: Trust region gradient projection for continual learning -- TRGP | 2022 | ICLR | `Opt-GradProj` |  |
| S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning -- S-Prompts | 2022 | NeurIPS | `Repr-Fix` | `TIL`, `DIL`, `CPT` |
| Continual learning with recursive gradient optimization -- RGO | 2022 | ICLR | `Opt-GradProj` |  |
| The challenges of continuous self-supervised learning -- MinRed | 2022 | ECCV | `Repr-SSL` |  |
| Towards better plasticity-stability trade-off in incremental learning: A simple linear connector -- Linear Connector | 2022 | CVPR | `Opt-Loss` |  |
| Learning to prompt for continual learning -- L2P | 2022 | CVPR | `Repr-Fix` | `TIL`, `CIL`, `DIL`, `CPT` |
| Generative negative text replay for continual vision-language pretraining (incCLIP) -- IncCLIP | 2022 | ArXiv | `Repr-ARL` | `CPT` |
| Incremental meta-learning via indirect discriminant alignment (IDA) -- IDA | 2022 | ArXiv | `Repr-ARL` | `CPT` |
| Helpful or harmful: Inter-task association in continual learning -- H2 | 2022 | ECCV | `Arch-Mask` |  |
| Foster: Feature boosting and compression for class-incremental learning -- FOSTER | 2022 | CVPR | `Rep-Feat` | `CIL` |
| Forward compatible few-shot class-incremental learning -- FACT | 2022 | CVPR | `Repr-Proto` | `FSCL` |
| Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima (f2m) -- F2M | 2022 | ArXiv | `Repr-Fix` | `FSCL` |
| Energy-based models for continual learning -- EBM-CL | 2022 | PMLR | `Repr-EBM` |  |
| DYTOX: Transformers for continual learning with DYnamic TOken EXpansion -- DYTOX | 2022 | CVPR | `Repr-Fix`, `Arch-Decomp` | `TIL` |
| DualPrompt: Complementary prompting for rehearsal-free continual learning -- DualPrompt | 2022 | ECCV | `Repr-Fix` | `CIL`, `CPT` |
| DLCFT: Deep linear continual fine-tuning for general incremental learning -- DLCFT | 2022 | ArXiv | `Repr-Fix` |  |
| Mimicking the oracle: An initial phase decorrelation approach for class incremental learning -- CwD | 2022 | CVPR | `Repr-Fix` |  |
| CoSCL: Cooperation of small continual learners is stronger than a big one -- CoSCL | 2022 | ECCV | `Arch-Mod` |  |
| Self-supervised models are continual learners -- CaSSLe | 2022 | CVPR | `Repr-SSL` | `TIL`, `CIL`, `DIL` |
| Anti-retroactive interference for lifelong learning -- ARI | 2022 | ECCV | `Opt-Meta` |  |
| Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation -- AFC | 2022 | CVPR | `KD-Feat` | `CIL` |
| Balancing stability and plasticity through advanced null space in continual learning -- AdNS | 2022 | ArXiv | `Opt-GradProj` |  |
| Model zoo: A growing “brain” that learns continually -- Zoo | 2021 | NeurIPS | `Arch-Mod` |  |
| Ss-il: Separated softmax for incremental learning -- SS-IL | 2021 | CVPR | `KD-Logit`, `Rep-Label` | `CIL` |
| Posterior meta-replay for continual learning -- PR | 2021 | NeurIPS | `Opt-Meta` |  |
| Insights from the future for continual learning -- PODNet | 2021 | CVPR | `KD-Feat`, `Rep-Feat` | `TIL`, `CIL` |
| Meta-learning with less forgetting on large-scale non-stationary task distributions (ORDER) -- ORDER | 2021 | ArXiv | `Repr-ARL` | `CPT` |
| Natural continual learning: Success is a journey, not (just) a destination -- NCL | 2021 | NeurIPS | `Opt-GradProj` |  |
| Efficient continual learning with modular networks and task-driven priors -- MNTDP | 2021 | ICLR | `Arch-Mod` |  |
| Linear mode connectivity in multitask and continual learning -- MC-SGD | 2021 | ICLR | `Opt-Loss` |  |
| Optimizing reusable knowledge for continual learning via metalearning -- MARK | 2021 | NeurIPS | `Opt-Meta`, `Arch-Decomp` |  |
| Continual learning via local module composition -- LMC | 2021 | NeurIPS | `Arch-Mod` |  |
| Using hindsight to anchor past knowledge in continual learning -- HAL | 2021 | AAAI | `Rep-Data` |  |
| Gradient projection memory for continual learning -- GPM | 2021 | ArXiv | `Opt-GradProj` |  |
| Few-shot lifelong learning -- FSLL | 2021 | AAAI | `Repr-Proto` | `FSCL` |
| Flattening sharpness for dynamic gradient projection memory benefits continual learning -- FS-DGPM | 2021 | NeurIPS | `Opt-GradProj` |  |
| Maintaining discrimination and fairness in class incremental learning -- WA | 2020 | CVPR | `Rep-Label` | `CIL` |
| Supermasks in superposition -- SupSup | 2020 | NeurIPS | `Opt-GradProj` | `TIL`, `CIL` |
| Understanding the role of training regimes in continual learning -- Stable-SGD | 2020 | NeurIPS | `Opt-Loss` |  |
| Side-tuning: A baseline for network adaptation via additive side networks -- Side-Tuning | 2020 | ECCV | `Repr-Fix` |  |
| Remind your neural network to prevent catastrophic forgetting -- REMIND | 2020 | ECCV | `Gen-Feat` | `TIL` |
| Online fast adaptation and knowledge accumulation (OSAKA): A new approach to continual learning -- OSAKA | 2020 | NeurIPS | `Opt-Meta` |  |
| Continual learning in low-rank orthogonal subspaces -- OrthogSubspace | 2020 | NeurIPS | `Opt-GradProj` |  |
| Orthogonal gradient descent for continual learning -- OGD | 2020 | AISTATS | `Opt-GradProj` |  |
| Mnemonics training: Multi-class incremental learning without forgetting -- Mnemonics | 2020 | CVPR | `Rep-Data`, `Rep-Feat`, `Gen-Data` |  |
| Merlin: Meta-consolidation for continual learning -- MERLIN | 2020 | NeurIPS | `Opt-Meta` | `OCL` |
| Representational continuity for unsupervised continual learning -- LUMP | 2020 | ArXiv | `Repr-SSL` |  |
| Look-ahead meta learning for continual learning -- La-MAML | 2020 | NeurIPS | `Opt-Meta` |  |
| iTAML: An incremental task-agnostic meta-learning approach -- iTAML | 2020 | CVPR | `Opt-Meta` |  |
| A neural dirichlet process mixture model for task-free continual learning -- GRU-D | 2020 | ICLR | `Repr-Gen` |  |
| Generalized variational continual learning -- GVCL | 2020 | ArXiv | `Arch-Decomp` |  |
| Gdumb: A simple approach that questions our progress in continual learning -- GDumb | 2020 | ECCV | `Rep-Data` | `TIL` |
| GAN memory with no forgetting -- GAN-memory | 2020 | NeurIPS | `Repr-Fix` |  |
| Class-incremental learning via deep model consolidation -- DMC | 2020 | CVPR | `KD-Rel` |  |
| Dark experience for general continual learning: A strong, simple baseline -- DER | 2020 | NeurIPS | `KD-Logit`, `KD-Proto`, `Rep-Data`, `Rep-Feat`, `Rep-Label`, `Repr-Proto` | `CIL` |
| Learning to continually learn -- ANML | 2020 | ECAI | `Opt-Meta` |  |
| Continual learning with node-importance based adaptive group sparse regularization -- AGS-CL | 2020 | NeurIPS | `Arch-Realloc` |  |
| Adversarial continual learning -- ACL | 2020 | ECCV | `Arch-Decomp` |  |
| Uncertainty-based continual learning with adaptive regularization -- UCL | 2019 | NeurIPS | `Arch-Realloc` |  |
| Random path selection for continual learning -- RPSNet | 2019 | NeurIPS | `Arch-Mod` |  |
| Rotate your networks: Better weight consolidation and less catastrophic forgetting -- R-EWC | 2019 | ICPR | `WR` |  |
| Prototype augmentation and self-supervised -- PASS | 2019 | CVPR | `Rep-Feat` |  |
| Continual learning of context-dependent processing in neural networks -- OWM | 2019 | Nature | `Opt-GradProj` | `CIL` |
| Meta-learning representations for continual learning -- OML | 2019 | ICML | `Opt-Meta` | `OCL` |
| Online continual learning with maximal interfered retrieval -- MIR | 2019 | NeurIPS | `Rep-Data` | `OCL`, `TFCL` |
| Il2M: Class incremental learning with dual memory -- IL2M | 2019 | ICCV | `Gen-Feat` | `CIL` |
| Learning without memorizing -- LWM | 2019 | CVPR | `WR` | `CIL` |
| Learning a unified classifier incrementally via rebalancing -- LUC | 2019 | CVPR | `WR`, `Rep-Feat` | `CIL` |
| Continual learning with hypernetworks -- HNET | 2019 | ArXiv | `WR`, `Gen-Data` | `TIL`, `CIL`, `TFCL` |
| Gradient based sample selection for online continual learning -- GSS | 2019 | NeurIPS | `Rep-Data` | `CIL`, `OCL`, `TFCL`, `FSCL` |
| Learning to remember: A synaptic plasticity driven framework for continual learning -- DGMa/DGMw/DGM | 2019 | CVPR | `Gen-Data`, `Arch-Seg` | `TIL` |
| Continual learning via neural pruning -- CLNP | 2019 | ArXiv | `Arch-Realloc` | `TIL` |
| Large scale incremental learning -- BiC | 2019 | CVPR | `Rep-Label` | `CIL` |
| Riemannian walk for incremental learning: Understanding forgetting and intransigence -- RWalk | 2018 | ECCV | `WR`, `Rep-Data` | `TIL`, `CIL` |
| Piggyback: Adapting a single network to multiple tasks by learning to mask weights -- Piggyback | 2018 | ECCV | `Arch-Mask` | `TIL` |
| PackNet: Adding multiple tasks to a single network by iterative pruning -- Packnet | 2018 | CVPR | `Arch-Realloc` | `TIL` |
| Memory replay GANs: Learning to generate images from new categories without forgetting -- MeRGANs | 2018 | NeurIPS | `Gen-Data` | `TIL` |
| Learning to learn without forgetting by maximizing transfer and minimizing interference -- MER | 2018 | ArXiv | `Opt-Meta` | `OCL` |
| Memory aware synapses: Learning what (not) to forget -- MAS | 2018 | ECCV | `WR` | `TIL`, `OCL`, `TFCL` |
| Overcoming catastrophic forgetting with hard attention to the task -- HAT | 2018 | ICML | `Arch-Mask` | `TIL` |
| End-to-end incremental learning -- EEIL | 2018 | ECCV | `KD-Logit`, `Rep-Label` | `CIL` |
| Encoder based lifelong learning -- EBLL | 2018 | ICCV | `KD-Feat` |  |
| Lifelong learning with dynamically expandable networks -- DEN | 2018 | ICLR | `Arch-Mod` | `TIL` |
| Progress & compress: A scalable framework for continual learning -- P&C | 2018 | TMLR | `WR` | `TIL` |
| On efficient lifelong learning with a-GEM -- A-GEM | 2018 | ArXiv | `Rep-Label`, `Opt-GradProj` | `TFCL` |
| Variational continual learning -- VCL | 2017 | ArXiv | `WR` | `TIL` |
| Continual learning through synaptic intelligence -- SI | 2017 | PMLR | reg; | `TIL` |
| PathNet: Evolution channels gradient descent in super neural networks -- PathNet | 2017 | ArXiv | `Arch-Mod` | `TIL` |
| Learning without forgetting -- LwF | 2017 | TPAMI | `KD-Logit`, `Rep-Label` | `TIL` |
| Gradient episodic memory for continual learning -- GEM | 2017 | NeurIPS | `Rep-Label`, `Opt-GradProj` | `TIL` |
| Overcoming catastrophic forgetting by incremental moment matching -- IMM | 2017 | NeurIPS | `WR` | `TIL` |
| iCaRL: Incremental classifier and representation learning -- iCaRL | 2017 | CVPR | `Rep-Data`, `Rep-Label`, `Repr-Proto` | `CIL` |
| FearNet: Brain-inspired model for incremental learning -- FearNet | 2017 | ArXiv | `Gen-Data` | `TIL` |
| Overcoming catastrophic forgetting in neural networks -- EWC | 2017 | PNAS | `WR` | `TIL` |
| Expert gate: Lifelong learning with a network of experts -- EG | 2017 | CVPR | `KD-Rel`, `Arch-Mod` |  |
| Continual learning with deep generative replay -- DGR | 2017 | NeurIPS | `Gen-Data` | `TIL` |
| Progressive neural networks -- PNN | 2016 | ArXiv | `Arch-Mod` |  |

---

## CL in Time Series

### Setting Taxonomy

- **Other CL Application**
  - Time Series Classification — `TSC`
  - Time Series Forecasting — `TSF`

### Research Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Knowledge informed time series forecasting -- KI-TSF | 2025 | KDD |  | `TSF` |
| IN-Flow: Instance Normalization Flow for Non-stationary Time Series Forecasting -- IN-Flow | 2025 | KDD |  | `TSF` |
| Fast and slow streams for online time series forecasting without information leakage -- DSOF | 2025 | ICLR |  | `TSF` |
| Distribution-aware online learning for urban spatiotemporal forecasting on streaming data -- DOL | 2025 | IJCAI |  | `TSF` |
| ODEStream: A buffer-free online learning framework with ode-based adaptor for streaming time series forecasting -- ODEStream | 2024 | ArXiv |  | `TSF` |
| A unified replay-based continuous learning framework for spatio-temporal prediction on streaming data -- STSimSiam | 2024 | ICDE |  | `TSF` |
| Temporal Continual Learning with Prior Compensation for Human Motion Prediction -- PCF | 2023 | NeurIPS |  | `TSC` |
| Pattern expansion and consolidation on evolving graphs for continual traffic prediction -- PECPM | 2023 | KDD |  | `TSF` |
| Online adaptive multivariate time series forecasting -- MTS | 2023 | KDD |  | `TSF` |
| Learning fast and slow for online time series forecasting -- FSNet | 2023 | ICLR |  | `TSF` |
| Futures quantitative investment with heterogeneous continual graph neural network -- HCGNN | 2023 | ArXiv |  | `TSF` |
| Dish-TS: A general paradigm for alleviating distribution shift in time series forecasting --Dish-TS | 2023 | AAAI |  | `TSF` |
| Streaming traffic flow prediction based on continuous reinforcement learning -- InTrans | 2022 | ICDM |  | `TSF` |
| Continual learning for human state monitoring - RNN | 2022 | ArXiv |  | `TSC` |
| TrafficStream: A streaming traffic flow forecasting framework based on graph neural networks and continual learning | 2021 | IJCAI |  | `TSF` |
| Spatio-temporal event forecasting using incremental multi-source feature learning -- HIML | 2021 | KDD |  | `TSF` |
| Continual learning for multivariate time series tasks with variable input dimensions -- IG | 2021 | ICDM |  | `TSC`, `TSF` |
| Continual learning augmented investment decisions -- CLA | 2018 | NeurIPS |  | `TSF` |

---

## CL in Natural Language Processing

### Research Papers

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

### Research Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Online Continual Learning For Interactive Instruction Following Agents -- CAMA | 2024 | ArXiv |  | `OCl` `BEIL` |
| Reinforced continual learning -- RCL | 2018 | NeurIPS |  | `TIL` |

---

## CL in Multimodal

### Survey Papers

| **Title** | **Year** | **Venue** | **Type** | **Setting** |
|-----------|----------|-----------|----------|-------------|
| AVQACL: A novel benchmark for audio-visual question answering continual learning -- AVQACL | 2025 | CVPR | benchmark paper | `TIL` |
| A practitioner’s guide to continual multimodal pretraining -- FoMo-in-Flux | 2024 | NeurIPS | benchmark paper | `CPT` |

---

## Acknowledgements
Inspired by awesome lists in continual learning and lifelong learning research.
