# Awesome General Continual Learning (CL)

A curated and structured list of **Continual Learning**

---

## Table of Contents
- [CL in Vision](#cl-in-vision)
  - [Method Taxonomy](#method-taxonomy)
  - [Setting Taxonomy](#setting-taxonomy)
  - [Survey Papers](#survey-papers)
  - [Research Papers](#research-papers)
- [CL in Time Series](#cl-in-time-series)
  - [Survey Papers](#survey-papers-1)
  - [Research Papers](#research-papers-1)
- [CL in Natural Language Processing](#cl-in-natural-language-processing)
  - [Survey Papers](#survey-papers-2)
  - [Research Papers](#research-papers-2)
- [CL in Reinforcement Learning](#cl-in-reinforcement-learning)
  - [Survey Papers](#survey-papers-3)
  - [Research Papers](#research-papers-3)
- [CL in Multimodal](#cl-in-multimodal)
  - [Survey Papers](#survey-papers-4)
  - [Research Papers](#research-papers-4)
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
    - Generative Classifier - `Gen-Class`

- **Representation-based**
  - Self-supervised learning — `Repr-SSL`
  - Pre-training for Downstream Tasks
    - Fixed Backbone — `Repr-Fix`
    - Updatable Backbone — `Repr-Upd`
  - Adaptive Representation Learning — `Repr-ARL`
  - Template-based Classification
    - Prototype-based — `Repr-Proto`
    - Generative — `Repr-Gen`
    - Energy-based — `Repr-EBM`

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
| A reality check on pre-training for exemplar-free class-incremental learning | 2025 | CVPR | benchmark paper |  |
| Continual learning: A systematic literature review | 2025 | Neural Networks | survey paper | `**` |
| A comprehensive survey of continual learning: theory, method and application | 2024 | TPAMI | survey paper |  |
| Class-incremental learning: A survey | 2024 | TPAMI | survey paper |  |
| Catastrophic forgetting in deep learning: A comprehensive taxonomy | 2024 | ArXiv | survey paper |  |
| Recent advances of continual learning in computer vision: An overview | 2024 | ArXiv | survey paper |  |
| Continual learning with pre-trained models: a survey | 2024 | IJCAI | survey paper | `**` |

### Research Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| C-loRA: Contextual low-rank adaptation for uncertainty estimation in large language models -- C-LoRA | 2025 | ArXiv | `Repr-ARL` |  |
| Componential prompt-knowledge alignment for domain incremental learning -- KA-Prompt | 2025 | ArXiv | `Repr-ARL` | `DIL` |
| Gated integration of low-rank adaptation for continual learning of language models -- GainLoRA | 2025 | ArXiv | `Repr-ARL` |  |
| RaSA: Rank-sharing low-rank adaptation -- RaSA | 2025 | ArXiv | `WR`, `Repr-ARL` |  |
| TreeloRA: Efficient continual learning via layer-wise loRAs guided by a hierarchical gradient-similarity tree -- TreeLoRA | 2025 | ArXiv | `Repr-ARL` |  |
| SD-LoRA: Scalable decoupled low-rank adaptation for class incremental learning -- SD-LoRA | 2025 | ArXiv | `Repr-ARL` |  |
| Adapter merging with centroid prototype mapping for scalable class-incremental learning -- ACMap | 2025 | CVPR | `Repr-Proto`, `**` |  |
| BiloRA: Almost-orthogonal parameter spaces for continual learning -- BiLoRA | 2025 | CVPR | `Repr-ARL` |  |
| CL-LoRA: Continual low-rank adaptation for rehearsal-free class-incremental learning -- CL-LoRA | 2025 | CVPR | `Repr-ARL` | `CIL` |
| Dual consolidation for pre-trained model-based domain-incremental learning -- DUCT  | 2025 | CVPR |  | `DIL` |
| LoRA subtraction for drift-resistant space in exemplar-free continual learning -- DRS | 2025 | CVPR | `Repr-ARL` |  |
| ProtoDepth: Unsupervised continual depth completion with prototypes -- ProtoDepth | 2025 | CVPR | `Repr-Proto` |  |
| Prototype augmented hypernetworks for continual learning -- PAH | 2025 | CVPR | `**` |  |
| Unsupervised continual domain shift learning with multi-prototype modeling -- UCDSL/MPM | 2025 | CVPR | `Repr-Proto` | `DIL` |
| Prototype antithesis for biological few-shot class-incremental learning -- PA | 2025 | ICLR | `Repr-Proto` | `FSCL` |
| Autoencoder-Based Hybrid Replay for Class-Incremental Learning -- HAE | 2025 | ICML | `Gen-Class` |  |
| Class incremental learning with self-supervised pre-training and prototype learning -- IPC | 2025 | Pattern Recognition | `**` |  |
| Contrastive continual learning with importance sampling and prototype-instance relation distillation -- CLIS | 2024 | AAAI | `**` |  |
| eTag: Class-incremental learning via embedding distillation and task-oriented generation -- eTag | 2024 | AAAI | `Gen-Class` |  |
| Fine-grained knowledge selection and restoration for non-exemplar class incremental learning -- FGKSR | 2024 | AAAI | `KD-Patch`, `KD-Proto` | `CIL`, `TFCL` |
| Controlled low-rank adaptation with subspace regularization for continued training on large language models -- CLoRA | 2024 | ArXiv | `WR`, `Repr-ARL` |  |
| Locality sensitive sparse encoding for learning world models online -- Losse-FTL | 2024 | ArXiv | `Repr-ARL` | `OCL` |
| PL-FSCIL: Harnessing the power of prompts for few-shot class-incremental learning -- PL-FSCIL | 2024 | ArXiv |  | `FSCL` |
| Expandable subspace ensemble for pre-trained model-based class-incremental learning -- EASE | 2024 | CVPR | `**` |  |
| Long-tail class incremental learning via independent sub-prototype construction -- SS | 2024 | CVPR | `**` |  |
| Resurrecting old classes with new data for exemplar-free continual learning -- ADC | 2024 | CVPR | `**` |  |
| Exemplar-free continual representation learning via learnable drift compensation -- LDC | 2024 | ECCV | `**` |  |
| A probabilistic framework for modular continual learning -- PICLE | 2024 | ICLR | `Arch-Mod` |  |
| Elastic feature consolidation for cold start exemplar-free incremental learning -- EFC | 2024 | ICLR | `KD-Feat` | `CIL` |
| Brain-inspired fast-and slow-update prompt tuning for few-shot class-incremental learning -- FSPT-FSCIL | 2024 | Neural Network |  | `FSCL` |
| Task confusion and catastrophic forgetting in class-incremental learning: A mathematical framework for discriminative and generative modelings | 2024 | NeurIPS | `Gen-Class` |  |
| Introspective GAN: Learning to grow a GAN for incremental generation and classification -- IntroGAN | 2024 | Pattern Recognition | `**` |  |
| Steering Prototypes with Prompt-tuning for Rehearsal-free Continual Learning -- CPP | 2024 | WACV | `**` |  |
| Decorate the newcomers: Visual domain prompt for continual test time adaptation -- VDP | 2023 | AAAI | `Repr-Fix` | `DIL` |
| Consistent prototype learning for few-shot continual relation extraction -- ConPL | 2023 | ACL | `Repr-Proto` | `FSCL` |
| Continual SLAM: Beyond lifelong simultaneous localization and mapping through continual learning -- CL-SLAM | 2023 | ArXiv | `Repr-SSL` |  |
| CODA-Prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning -- CODA-Prompt | 2023 | CVPR | `Repr-Fix` | `CIL`, `CPT` |
| EcoTTA: Memory-efficient continual test-time adaptation via self-distilled regularization -- EcoTTA | 2023 | CVPR | `KD-Feat` |  |
| FeTrIL: Feature translation for exemplar-free class-incremental learning -- FeTril | 2023 | CVPR | `Repr-Fix`, `**` |  |
| GKEAL: Gaussian kernel embedded analytic learning for few-shot class incremental task -- GKEAL | 2023 | CVPR |  | `FSCL` |
| PCR: Proxy-based Contrastive Replay for Online Class-Incremental Continual Learning -- PCR  | 2023 | CVPR |  | `OCL` | 
| PIVOT: Prompting for video continual learning -- PIVOT | 2023 | CVPR | `Repr-Fix` | `CIL` |
| Few-shot continual infomax learning -- FCIL | 2023 | ICCV |  | `FSCL` |
| NAPA-VQ: Neighborhood aware prototype augmentation with vector quantization for continual learning -- NAPA-VQ | 2023 | ICCV | `**` |  |
| Online prototype learning for online continual learning -- OnPro | 2023 | ICCV | `**` |  |
| Prototype reminiscence and augmented asymmetric knowledge aggregation for non-exemplar class-incremental learning -- PR/AKA | 2023 | ICCV | `**` |  |
| SLCA: Slow learner with classifier alignment for continual learning on a pre-trained model -- SLCA | 2023 | ICCV | `Repr-Fix`, `**` |  |
| Continual momentum filtering on parameter space for online test-time adaptation -- CMF | 2023 | ICLR | `WR` | `OCL` |
| Kalman filter online learning from non-stationary data -- KFOCL | 2023 | ICLR | `Repr-Upd` | `OCL`, `CPT` |
| Progressive prompts: Continual learning for language models -- Progressive-Prompts | 2023 | ICLR | `Repr-Fix` | `CPT` |
| Prototype-sample relation distillation: towards replay-free continual learning -- PRD | 2023 | ICML | `**` |  |
| Revisiting class-incremental learning with pre-trained models: generalizability and adaptivity are all you need -- APER | 2023 | IJCV | `**` |  |
| An empirical investigation of the role of pre-training in lifelong learning -- SAM | 2023 | JMLR | `Repr-Upd`, `Opt-Meta`, `Opt-Loss` | `CPT` |
| Incorporating neuro-inspired adaptability for continual learning in artificial intelligence -- CAF | 2023 | Nature | `Arch-Mod` |  |
| FeCAM: Exploiting the heterogeneity of class distributions in exemplar-free continual learning -- FeCAM | 2023 | NeurIPS | `**` |  |
| Few-shot class-incremental learning via training-free prototype calibration -- TEEN | 2023 | NeurIPS | `Repr-Proto` | `FSCL` |
| RanPAC: Random projections and pre-trained models for continual learning -- RanPAC | 2023 | NeurIPS | `**` |  |
| Balancing stability and plasticity through advanced null space in continual learning -- AdNS | 2022 | ArXiv | `Opt-GradProj` |  |
| DLCFT: Deep linear continual fine-tuning for general incremental learning -- DLCFT | 2022 | ArXiv | `Repr-Fix` |  |
| Generative negative text replay for continual vision-language pretraining (incCLIP) -- IncCLIP | 2022 | ArXiv | `Repr-ARL` | `CPT` |
| Incremental meta-learning via indirect discriminant alignment (IDA) -- IDA | 2022 | ArXiv | `Repr-ARL` | `CPT` |
| Overcoming catastrophic forgetting in incremental few-shot learning by finding flat minima (f2m) -- F2M | 2022 | ArXiv | `Repr-Fix` | `FSCL` |
| Class-Incremental Learning by Knowledge Distillation with Adaptive Feature Consolidation -- AFC | 2022 | CVPR | `KD-Feat` | `CIL` |
| Constrained few-shot class-incremental learning -- CFSCIL | 2022 | CVPR |  | `FSCL` |
| DYTOX: Transformers for continual learning with DYnamic TOken EXpansion -- DYTOX | 2022 | CVPR | `Repr-Fix`, `Arch-Decomp` | `TIL` |
| Few-shot class-incremental learning via feature space composition -- FRoST | 2022 | CVPR |  | `FSCL` |
| Forward compatible few-shot class-incremental learning -- FACT | 2022 | CVPR | `Repr-Proto` | `FSCL` |
| Foster: Feature boosting and compression for class-incremental learning -- FOSTER | 2022 | CVPR | `Rep-Feat` | `CIL` |
| Learning to prompt for continual learning -- L2P | 2022 | CVPR | `Repr-Fix` | `TIL`, `CIL`, `DIL`, `CPT` |
| Mimicking the oracle: An initial phase decorrelation approach for class incremental learning -- CwD | 2022 | CVPR | `Repr-Fix`, `**` |  |
| Probing representation forgetting in supervised and unsupervised continual learning -- Probe | 2022 | CVPR | `**` |  |
| Rainbow Memory: Continual Learning with a Memory of Diverse Samples -- RM  | 2022 | CVPR |  | `CIL` |
| Self-supervised models are continual learners -- CaSSLe | 2022 | CVPR | `Repr-SSL` | `TIL`, `CIL`, `DIL` |
| Self-sustaining representation expansion for non-exemplar class-incremental learning -- SSRE | 2022 | CVPR | `**` |  |
| Self-supervised stochastic classifier for few-shot class incremental learning -- S3C | 2022 | CVPR |  | `FSCL` |
| Towards better plasticity-stability trade-off in incremental learning: A simple linear connector -- Linear Connector | 2022 | CVPR | `Opt-Loss` |  |
| Anti-retroactive interference for lifelong learning -- ARI | 2022 | ECCV | `Opt-Meta` |  |
| CoSCL: Cooperation of small continual learners is stronger than a big one -- CoSCL | 2022 | ECCV | `Arch-Mod` |  |
| DualPrompt: Complementary prompting for rehearsal-free continual learning -- DualPrompt | 2022 | ECCV | `Repr-Fix` | `CIL`, `CPT` |
| Helpful or harmful: Inter-task association in continual learning -- H2 | 2022 | ECCV | `Arch-Mask` |  |
| The challenges of continuous self-supervised learning -- MinRed | 2022 | ECCV | `Repr-SSL` |  |
| Transfer without forgetting -- TwF | 2022 | ECCV | `Repr-Fix` |  |
| Continual learning with recursive gradient optimization -- RGO | 2022 | ICLR | `Opt-GradProj` |  |
| TRGP: Trust region gradient projection for continual learning -- TRGP | 2022 | ICLR | `Opt-GradProj` |  |
| Forget-free continual learning with winning subnetworks -- WSN | 2022 | ICML | `Arch-Mask` |  |
| S-prompts learning with pre-trained transformers: An occam’s razor for domain incremental learning -- S-Prompts | 2022 | NeurIPS | `Repr-Fix` | `TIL`, `DIL`, `CPT` |
| Energy-based models for continual learning -- EBM-CL | 2022 | PMLR | `Repr-EBM` |  |
| Class-incremental continual learning into the extended der-verse -- X-DER | 2022 | TPAMI | `Rep-Data` |  |
| Few-shot lifelong learning -- FSLL | 2021 | AAAI | `Repr-Proto` | `FSCL` |
| Using hindsight to anchor past knowledge in continual learning -- HAL | 2021 | AAAI | `Rep-Data` |  |
| Co-transport for class-incremental learning -- COIL | 2021 | ACL | `**` |  |
| Gradient projection memory for continual learning -- GPM | 2021 | ArXiv | `Opt-GradProj` |  |
| Memory efficient continual learning with transformers (ADA) -- ADA | 2021 | ArXiv | `Repr-Fix` |  |
| Meta-learning with less forgetting on large-scale non-stationary task distributions (ORDER) -- ORDER | 2021 | ArXiv | `Repr-ARL` | `CPT` |
| New insights on reducing abrupt representation change in online continual learning -- ER-ACE/ER-AML | 2021 | ArXiv | `Rep-Embed` | `OCL` |
| Adaptive aggregation networks for class-incremental learning -- AANets | 2021 | CVPR | `Rep-Feat` |  |
| Class-Incremental Learning with Generative Classifiers -- GC | 2021 | CVPR | `Gen-Class` |  |
| Continual adaptation of visual representations via domain randomization and meta-learning -- Meta-DR  | 2021 | CVPR |  | `DIL` |
| DER: Dynamically expandable representation for class incremental learning -- DER | 2021 | CVPR | `**` |  |
| Distilling causal effect of data in class-incremental learning -- DDE | 2021 | CVPR | `Rep-Feat` |  |
| Few-shot class-incremental learning via continually evolved classifiers -- CEC | 2021 | CVPR | `Repr-Proto` |  |
| Insights from the future for continual learning -- PODNet | 2021 | CVPR | `KD-Feat`, `Rep-Feat`, `**` | `TIL`, `CIL` |
| Prototype augmentation and self-supervised -- PASS | 2021 | CVPR | `Rep-Feat`, `**` |  |
| Ss-il: Separated softmax for incremental learning -- SS-IL | 2021 | CVPR | `KD-Logit`, `Rep-Label` | `CIL` |
| Training networks in null space of feature covariance for continual learning -- AdamNSCL | 2021 | CVPR | `Opt-GradProj` |  |
| Co2l: Contrastive continual learning -- Co2L | 2021 | ICCV | `Rep-Feat`, `Repr-SSL` | `TIL`, `CIL`, `DIL` |
| Few-shot and continual learning with attentive independent mechanisms -- AIM | 2021 | ICCV | `Opt-Meta` | `FSCL` |
| Striking a balance between stability and plasticity for class-incremental learning -- SPB-I/SPB-M| 2021 | ICCV | `**` |  |
| GP-Tree: A hierarchical Gaussian process model for few-shot class-incremental learning -- GP-Tree | 2021 | ICML |  | `FSCL` |
| Efficient continual learning with modular networks and task-driven priors -- MNTDP | 2021 | ICLR | `Arch-Mod` |  |
| Linear mode connectivity in multitask and continual learning -- MC-SGD | 2021 | ICLR | `Opt-Loss` |  |
| BNS: Building network structures dynamically for continual learning -- BNS | 2021 | NeurIPS | `Arch-Seg` |  |
| Class-incremental learning via dual augmentation -- classAug | 2021 | NeurIPS | `**` |  |
| Continual learning via local module composition -- LMC | 2021 | NeurIPS | `Arch-Mod` |  |
| Flattening sharpness for dynamic gradient projection memory benefits continual learning -- FS-DGPM | 2021 | NeurIPS | `Opt-GradProj` |  |
| Model zoo: A growing “brain” that learns continually -- Zoo | 2021 | NeurIPS | `Arch-Mod` |  |
| Natural continual learning: Success is a journey, not (just) a destination -- NCL | 2021 | NeurIPS | `Opt-GradProj` |  |
| Optimizing reusable knowledge for continual learning via metalearning -- MARK | 2021 | NeurIPS | `Opt-Meta`, `Arch-Decomp` |  |
| Posterior meta-replay for continual learning -- PR | 2021 | NeurIPS | `Opt-Meta` |  |
| Orthogonal gradient descent for continual learning -- OGD | 2020 | AISTATS | `Opt-GradProj` |  |
| Generalized variational continual learning -- GVCL | 2020 | ArXiv | `Arch-Decomp` |  |
| Representational continuity for unsupervised continual learning -- LUMP | 2020 | ArXiv | `Repr-SSL` |  |
| Class-incremental learning via deep model consolidation -- DMC | 2020 | CVPR | `KD-Rel` |  |
| Few-shot class-incremental learning -- TOPIC  | 2020 | CVPR |  | `CIL`, `FSCL` |
| iTAML: An incremental task-agnostic meta-learning approach -- iTAML | 2020 | CVPR | `Opt-Meta` |  |
| Maintaining discrimination and fairness in class incremental learning -- WA | 2020 | CVPR | `Rep-Label` | `CIL` |
| Mnemonics training: Multi-class incremental learning without forgetting -- Mnemonics | 2020 | CVPR | `Rep-Data`, `Rep-Feat`, `Gen-Data` |  |
| Semantic drift compensation for class-incremental learning -- SDC | 2020 | CVPR | `**` |  |
| Learning to continually learn -- ANML | 2020 | ECAI | `Opt-Meta` |  |
| Adversarial continual learning -- ACL | 2020 | ECCV | `Arch-Decomp` |  |
| Gdumb: A simple approach that questions our progress in continual learning -- GDumb | 2020 | ECCV | `Rep-Data` | `TIL` |
| PODNet: Pooled outputs distillation for small-tasks incremental learning -- PODNet | 2020 | ECCV | `KD-Feat`, `Rep-Feat`, `**` | `TIL`, `CIL` |
| Remind your neural network to prevent catastrophic forgetting -- REMIND | 2020 | ECCV | `Gen-Feat` | `TIL` |
| Side-tuning: A baseline for network adaptation via additive side networks -- Side-Tuning | 2020 | ECCV | `Repr-Fix` |  |
| Continual prototype evolution: learning online from non-stationary data streams -- COPE/learner-evaluator | 2020 | ICCV | `**` |  |
| A neural dirichlet process mixture model for task-free continual learning -- GRU-D | 2020 | ICLR | `Repr-Gen` |  |
| Continual learning in low-rank orthogonal subspaces -- OrthogSubspace | 2020 | NeurIPS | `Opt-GradProj` |  |
| Continual learning with node-importance based adaptive group sparse regularization -- AGS-CL | 2020 | NeurIPS | `Arch-Realloc` |  |
| Dark experience for general continual learning: A strong, simple baseline -- DER | 2020 | NeurIPS | `KD-Logit`, `KD-Proto`, `Rep-Data`, `Rep-Feat`, `Rep-Label`, `Repr-Proto` | `CIL` |
| GAN memory with no forgetting -- GAN-memory | 2020 | NeurIPS | `Repr-Fix` |  |
| Look-ahead meta learning for continual learning -- La-MAML | 2020 | NeurIPS | `Opt-Meta` |  |
| Merlin: Meta-consolidation for continual learning -- MERLIN | 2020 | NeurIPS | `Opt-Meta` | `OCL` |
| Online fast adaptation and knowledge accumulation (OSAKA): A new approach to continual learning -- OSAKA | 2020 | NeurIPS | `Opt-Meta` |  |
| Supermasks in superposition -- SupSup | 2020 | NeurIPS | `Opt-GradProj` | `TIL`, `CIL` |
| Understanding the role of training regimes in continual learning -- Stable-SGD | 2020 | NeurIPS | `Opt-Loss` |  |
| Continual learning via neural pruning -- CLNP | 2019 | ArXiv | `Arch-Realloc` | `TIL` |
| Continual learning with hypernetworks -- HNET | 2019 | ArXiv | `WR`, `Gen-Data` | `TIL`, `CIL`, `TFCL` |
| Large scale incremental learning -- BiC | 2019 | CVPR | `Rep-Label` | `CIL` |
| Learning a unified classifier incrementally via rebalancing -- LUC | 2019 | CVPR | `WR`, `Rep-Feat`, `**` | `CIL` |
| Learn to grow: A continual structure learning framework for overcoming catastrophic forgetting -- L2G  | 2019 | ArXiv |  | `CIL` |
| Learning to remember: A synaptic plasticity driven framework for continual learning -- DGMa/DGMw/DGM | 2019 | CVPR | `Gen-Data`, `Arch-Seg` | `TIL` |
| Learning without memorizing -- LWM | 2019 | CVPR | `WR` | `CIL` |
| Il2M: Class incremental learning with dual memory -- IL2M | 2019 | ICCV | `Gen-Feat` | `CIL` |
| Meta-learning representations for continual learning -- OML | 2019 | ICML | `Opt-Meta` | `OCL` |
| Rotate your networks: Better weight consolidation and less catastrophic forgetting -- R-EWC | 2019 | ICPR | `WR` |  |
| Continual learning of context-dependent processing in neural networks -- OWM | 2019 | Nature | `Opt-GradProj` |  |
| Gradient based sample selection for online continual learning -- GSS | 2019 | NeurIPS | `Rep-Data` | `CIL`, `OCL`, `TFCL`, `FSCL` |
| Online continual learning with maximal interfered retrieval -- MIR | 2019 | NeurIPS | `Rep-Data` | `OCL`, `TFCL` |
| Random path selection for continual learning -- RPSNet | 2019 | NeurIPS | `Arch-Mod` |  |
| Uncertainty-based continual learning with adaptive regularization -- UCL | 2019 | NeurIPS | `Arch-Realloc` |  |
| Learning to learn without forgetting by maximizing transfer and minimizing interference -- MER | 2018 | ArXiv | `Opt-Meta` | `OCL` |
| On efficient lifelong learning with a-GEM -- A-GEM | 2018 | ArXiv | `Rep-Label`, `Opt-GradProj` | `TFCL` |
| PackNet: Adding multiple tasks to a single network by iterative pruning -- Packnet | 2018 | CVPR | `Arch-Realloc` | `TIL` |
| Rethinking feature distribution for loss functions in image classification -- L-GM | 2018 | CVPR | `Repr-Gen` |  |
| End-to-end incremental learning -- EEIL | 2018 | ECCV | `KD-Logit`, `Rep-Label` | `CIL` |
| Memory aware synapses: Learning what (not) to forget -- MAS | 2018 | ECCV | `WR` | `TIL`, `OCL`, `TFCL` |
| Piggyback: Adapting a single network to multiple tasks by learning to mask weights -- Piggyback | 2018 | ECCV | `Arch-Mask` | `TIL` |
| Riemannian walk for incremental learning: Understanding forgetting and intransigence -- RWalk | 2018 | ECCV | `WR`, `Rep-Data` | `TIL`, `CIL` |
| Encoder based lifelong learning -- EBLL | 2018 | ICCV | `KD-Feat`, `**` |  |
| Lifelong learning with dynamically expandable networks -- DEN | 2018 | ICLR | `Arch-Mod` | `TIL` |
| Overcoming catastrophic forgetting with hard attention to the task -- HAT | 2018 | ICML | `Arch-Mask` | `TIL` |
| Memory replay GANs: Learning to generate images from new categories without forgetting -- MeRGANs | 2018 | NeurIPS | `Gen-Data` | `TIL` |
| Progress & compress: A scalable framework for continual learning -- C/P&C | 2018 | TMLR | `WR` | `TIL` |
| FearNet: Brain-inspired model for incremental learning -- FearNet | 2017 | ArXiv | `Gen-Data` | `TIL` |
| PathNet: Evolution channels gradient descent in super neural networks -- PathNet | 2017 | ArXiv | `Arch-Mod` | `TIL` |
| Variational continual learning -- VCL | 2017 | ArXiv | `WR` | `TIL` |
| Expert gate: Lifelong learning with a network of experts -- EG/GATE | 2017 | CVPR | `KD-Rel`, `Arch-Mod` |  |
| iCaRL: Incremental classifier and representation learning -- iCaRL | 2017 | CVPR | `Rep-Data`, `Rep-Label`, `Repr-Proto`, `**` | `CIL` |
| Continual learning through synaptic intelligence -- SI | 2017 | ICML | reg; | `TIL` |
| Continual learning with deep generative replay -- DGR | 2017 | NeurIPS | `Gen-Data` | `TIL` |
| Gradient episodic memory for continual learning -- GEM | 2017 | NeurIPS | `Rep-Label`, `Opt-GradProj` | `TIL` |
| Overcoming catastrophic forgetting by incremental moment matching -- IMM | 2017 | NeurIPS | `WR` | `TIL` |
| Overcoming catastrophic forgetting in neural networks -- EWC | 2017 | PNAS | `WR` | `TIL` |
| Learning without forgetting -- LwF | 2017 | TPAMI | `KD-Logit`, `Rep-Label` | `TIL` |
| Progressive neural networks -- PNN | 2016 | ArXiv | `Arch-Mod` |  |

---

## CL in Time Series

### Setting Taxonomy

- **Other CL Application**
  - Time Series Classification — `TSC`
  - Time Series Forecasting — `TSF`

### Survey Papers

| **Title** | **Year** | **Venue** | **Type** | **Setting** |
|-----------|----------|-----------|----------|-------------|
| Class-incremental learning for time series: Benchmark and evaluation | 2024 | KDD | benchmark paper |  |

### Research Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| CA-MoE: Channel-Adapted MoE for Incremental Weather Forecasting -- CA-MoE | 2025 | ArXiv |  | `TSF` |
| VA-MoE: Variables-Adaptive mixture of experts for incremental weather forecasting -- VA-MoE | 2025 | CVPR |  | `TSF` |
| Fast and slow streams for online time series forecasting without information leakage -- DSOF | 2025 | ICLR |  | `TSF` |
| Distribution-aware online learning for urban spatiotemporal forecasting on streaming data -- DOL | 2025 | IJCAI |  | `TSF` |
| IN-Flow: Instance Normalization Flow for Non-stationary Time Series Forecasting -- IN-Flow | 2025 | KDD |  | `TSF` |
| Knowledge informed time series forecasting -- KI-TSF | 2025 | KDD |  | `TSF` |
| ODEStream: A buffer-free online learning framework with ode-based adaptor for streaming time series forecasting -- ODEStream | 2024 | ArXiv |  | `TSF` |
| A unified replay-based continuous learning framework for spatio-temporal prediction on streaming data -- STSimSiam | 2024 | ICDE |  | `TSF` |
| Dish-TS: A general paradigm for alleviating distribution shift in time series forecasting --Dish-TS | 2023 | AAAI |  | `TSF` |
| Futures quantitative investment with heterogeneous continual graph neural network -- HCGNN | 2023 | ArXiv |  | `TSF` |
| Learning fast and slow for online time series forecasting -- FSNet | 2023 | ICLR |  | `TSF` |
| Online adaptive multivariate time series forecasting -- MTS | 2023 | KDD |  | `TSF` |
| Pattern expansion and consolidation on evolving graphs for continual traffic prediction -- PECPM | 2023 | KDD |  | `TSF` |
| Temporal Continual Learning with Prior Compensation for Human Motion Prediction -- PCF | 2023 | NeurIPS |  | `TSC` |
| Continual learning for human state monitoring - RNN | 2022 | ArXiv |  | `TSC` |
| Streaming traffic flow prediction based on continuous reinforcement learning -- InTrans | 2022 | ICDM |  | `TSF` |
| Spatio-temporal event forecasting using incremental multi-source feature learning -- HIML | 2021 | KDD |  | `TSF` |
| Continual learning for multivariate time series tasks with variable input dimensions -- IG | 2021 | ICDM |  | `TSC`, `TSF` |
| TrafficStream: A streaming traffic flow forecasting framework based on graph neural networks and continual learning | 2021 | IJCAI |  | `TSF` |
| Continual learning augmented investment decisions -- CLA | 2018 | NeurIPS |  | `TSF` |

---

## CL in Natural Language Processing

### Research Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Knowledge decoupling via orthogonal projection for lifelong editing of large language models -- KDE | 2025 | ACL |  | `TFCL` |
| Neuron-level sequential editing for large language models -- NSE | 2025 | ACL |  | `TFCL` |
| Serial lifelong editing via mixture of knowledge experts -- ARM | 2025 | ACL |  | `TFCL` |
| HiDe-LLaVA: Hierarchical decoupling for continual instruction tuning of multimodal large language models -- hiDe-LLaVA | 2025 | ArXiv |  | `TFCL` |
| Continual pretraining of language models -- DAS | 2023 | ICLR |  | `CPT` |
| Lifelong language pretraining with distribution-specialized experts -- Lifelong-MoE | 2023 | ICML |  | `CPT` |
| Continual pre-training of language models for math problem understanding with syntax-aware memory network -- SNCL | 2022 | ACL |  | `CPT` |
| Learn continually, generalize rapidly: Lifelong knowledge accumulation for few-shot learning -- LKA-FSL | 2021 | EMNLP |  | `FSCL` |
| Continual relation learning across domains -- EMAR | 2020 | ACL |  | `DIL` |
| ERNIE 2.0: A continual pre-training framework for language understanding -- ERNIE2.0 | 2020 | AAAI |  | `CPT` |

---

## CL in Reinforcement Learning

### Survey Papers

| **Title** | **Year** | **Venue** | **Type** | **Setting** |
|-----------|----------|-----------|----------|-------------|
| A survey of continual reinforcement learning | 2025 | TPAMI | survey paper |  |

### Research Papers

| **Title** | **Year** | **Venue** | **CL Method** | **CL Setting** |
|-----------|----------|-----------|---------------|----------------|
| Online Continual Learning For Interactive Instruction Following Agents -- CAMA | 2024 | ArXiv |  | `OCl`, `BEIL` |
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
Inspired by awesome lists in continual learning:
