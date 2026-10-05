# Graph Report - Endometriosis-Identification-System-Using-a-Multi-Model-Voting-Ensemble-and-Explainable-AI  (2026-10-03)

## Corpus Check
- Corpus is ~2,334 words - fits in a single context window. You may not need a graph.

## Summary
- 116 nodes · 196 edges · 9 communities (6 shown, 3 thin omitted)
- Extraction: 90% EXTRACTED · 10% INFERRED · 0% AMBIGUOUS · INFERRED: 20 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Dataset Discovery and Splitting
- Ensemble Explainability Design
- TensorFlow Dataset Loading
- Data Configuration and Governance
- Slice Preprocessing
- Three-Channel Slice Loading
- Pipeline Orchestration

## God Nodes (most connected - your core abstractions)
1. `Python Machine Learning Software Stack` - 11 edges
2. `run_pipeline()` - 10 edges
3. `MRI Preprocessing Pipeline` - 10 edges
4. `Endometriosis Identification System` - 9 edges
5. `build_dataset()` - 8 edges
6. `preprocess_slice()` - 8 edges
7. `discover_patients()` - 7 edges
8. `create_train_val_datasets()` - 7 edges
9. `load_nifti()` - 7 edges
10. `Training and Data Configuration` - 7 edges

## Surprising Connections (you probably didn't know these)
- `Preprocessing Pipeline` --conceptually_related_to--> `MRI Preprocessing Pipeline`  [INFERRED]
  data/README.md → README.md
- `Local MRI Dataset Directory` --shares_data_with--> `Endometriosis Identification System`  [INFERRED]
  data/README.md → README.md
- `MRI Preprocessing Configuration` --conceptually_related_to--> `MRI Preprocessing Pipeline`  [INFERRED]
  configs/config.yaml → README.md
- `PyYAML` --conceptually_related_to--> `Training and Data Configuration`  [INFERRED]
  requirements.txt → configs/config.yaml
- `Augmentation Configuration` --conceptually_related_to--> `Data Augmentation`  [INFERRED]
  configs/config.yaml → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Transfer-Learned CNNs Form the Soft Voting Ensemble** — readme_transfer_learning, readme_resnet50, readme_efficientnetv2, readme_densenet121, readme_soft_voting_ensemble [EXTRACTED 1.00]

## Communities (9 total, 3 thin omitted)

### Community 0 - "Dataset Discovery and Splitting"
Cohesion: 0.12
Nodes (8): assign_label(), discover_patients(), get_sequence_files(), save_manifest(), split_patients(), run_pipeline(), extract_axial_slices(), load_nifti()

### Community 1 - "Ensemble Explainability Design"
Cohesion: 0.12
Nodes (25): Accuracy, Precision, Sensitivity, F1-Score, and AUC-ROC, Clinical Decision-Support Tool, DenseNet121, EfficientNetV2, Endometriosis, Endometriosis Identification System, Explainable Artificial Intelligence, Gradient-Weighted Class Activation Mapping (Grad-CAM) (+17 more)

### Community 2 - "TensorFlow Dataset Loading"
Cohesion: 0.18
Nodes (6): _assert_disjoint_patients(), build_dataset(), count_per_split(), create_train_val_datasets(), load_manifest(), build_augmentation_layer()

### Community 3 - "Data Configuration and Governance"
Cohesion: 0.15
Nodes (15): Augmentation Configuration, Data Loader Configuration, Healthy Pelvis Dataset, T1, T1FS, T2, and T2FS MRI Sequences, MRI Preprocessing Configuration, Training and Validation Split, Training and Data Configuration, UT-EndoMRI Dataset (+7 more)

### Community 4 - "Slice Preprocessing"
Cohesion: 0.25
Nodes (3): min_max_normalize(), preprocess_slice(), resize_slice()

### Community 5 - "Three-Channel Slice Loading"
Cohesion: 0.29
Nodes (3): _decode_map(), _load_and_expand(), to_three_channels()

## Knowledge Gaps
- **7 isolated node(s):** `Endometriosis`, `Accuracy, Precision, Sensitivity, F1-Score, and AUC-ROC`, `Clinical Decision-Support Tool`, `T1, T1FS, T2, and T2FS MRI Sequences`, `Data Loader Configuration` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 42 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Endometriosis Identification System` connect `Ensemble Explainability Design` to `Data Configuration and Governance`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Why does `MRI Preprocessing Pipeline` connect `Ensemble Explainability Design` to `Data Configuration and Governance`?**
  _High betweenness centrality (0.039) - this node is a cross-community bridge._
- **Why does `to_three_channels()` connect `Three-Channel Slice Loading` to `TensorFlow Dataset Loading`, `Slice Preprocessing`?**
  _High betweenness centrality (0.035) - this node is a cross-community bridge._
- **Are the 7 inferred relationships involving `MRI Preprocessing Pipeline` (e.g. with `MRI Preprocessing Configuration` and `Preprocessing Pipeline`) actually correct?**
  _`MRI Preprocessing Pipeline` has 7 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Endometriosis`, `Accuracy, Precision, Sensitivity, F1-Score, and AUC-ROC`, `Clinical Decision-Support Tool` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Dataset Discovery and Splitting` be split into smaller, more focused modules?**
  _Cohesion score 0.11904761904761904 - nodes in this community are weakly interconnected._
- **Should `Ensemble Explainability Design` be split into smaller, more focused modules?**
  _Cohesion score 0.12333333333333334 - nodes in this community are weakly interconnected._