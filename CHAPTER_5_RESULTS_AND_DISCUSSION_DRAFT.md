# CHAPTER 5

# RESULTS AND DISCUSSION

This chapter presents the completed internal evaluation of three transfer-learning models and their equal-weight soft-voting ensemble. It reports the data composition, analysis approach, training behavior, patient-level performance, model agreement, and exploratory Grad-CAM audit before interpreting the findings in relation to the literature and study design.

The reported results must be interpreted according to the operational task that was actually evaluated. The positive class consisted of records from the UT-EndoMRI endometriosis cohort, whereas the comparator class consisted of records from the MOGaMBO cervical-cancer cohort. MOGaMBO was not a healthy-control dataset, and its members were not clinically verified as endometriosis-negative. Because source cohort and operational label were completely associated, the experiment measured internal discrimination between these two cohorts rather than clinical diagnosis of endometriosis versus confirmed absence of disease. The hospital dataset was not included in the reported results because sequence verification, radiologist-assisted reference-label verification, and the applicable ethical determination remained pending.

## 5.1 Overview of the Data Analysis Approach

The internal experiment used T2-weighted pelvic MRI volumes from UT-EndoMRI and MOGaMBO. Following file discovery and exact voxel-content deduplication, 149 retained volumes associated with dataset patient identifiers were available for model development. These records generated 7,750 retained axial slices after orientation harmonization, spatial standardization, foreground-based slice filtering, intensity normalization, cropping or padding, and resizing to 224 x 224 pixels.

The data were partitioned at the dataset patient-identifier level using a fixed random seed of 2027 and stratified by source site. This produced mutually exclusive training, validation, and internal-test partitions while preserving representation from UT-D1, UT-D2, and MOGaMBO. Patient-level partitioning prevented correlated slices from the same volume from entering both development and evaluation sets. This follows the medical-imaging workflow described by Montagnon et al. (2020), in which the independent test partition is reserved until model development is complete.

ResNet50, EfficientNetV2S, and DenseNet121 were initialized using ImageNet-pretrained weights. The pretrained backbones remained frozen, while a dropout layer and a one-unit sigmoid classification head were trained for each architecture. Slice probabilities were averaged within each patient identifier to generate one patient-level probability per model. The soft-voting ensemble probability was then calculated as the arithmetic mean of the three patient-level probabilities. A fixed threshold of 0.5 was used for all binary decisions; the threshold and voting weights were not optimized after viewing the internal test results.

Accuracy, balanced accuracy, sensitivity, specificity, precision, F1-score, ROC-AUC, and Brier score were calculated at the patient-record level. Wilson 95% confidence intervals were used for accuracy, sensitivity, specificity, and precision. A retained 2,000-iteration class-stratified patient bootstrap using seed 2027 supplied percentile intervals for the other metrics. Agreement was assessed using pairwise overall agreement, Cohen's kappa, probability correlations, and voting patterns. No formal paired test of model superiority was conducted.

## 5.2 Presentation of Findings

### 5.2.1 Dataset Composition

Table 9 summarizes the realized patient-record and slice counts. The training partition contained 104 records and 5,400 slices, the validation partition contained 22 records and 1,063 slices, and the internal-test partition contained 23 records and 1,287 slices. The term record is used because the available public metadata did not permit independent confirmation that every retained dataset identifier represented a different physical person.

**Table 9**  
*Composition of the Internal Development and Evaluation Partitions*

| Partition | Records | UT operational-positive records | MOGaMBO proxy-comparator records | Retained slices |
|---|---:|---:|---:|---:|
| Training | 104 | 63 | 41 | 5,400 |
| Validation | 22 | 13 | 9 | 1,063 |
| Internal test | 23 | 14 | 9 | 1,287 |
| **Total** | **149** | **90** | **59** | **7,750** |

The source distribution was 45 UT-D1 records, 45 UT-D2 records, and 59 MOGaMBO records. Before model-visible geometry was harmonized, a baseline classifier using only original volume dimensions and voxel spacing achieved an accuracy of 0.9597 and a ROC-AUC of 0.9652. When the same type of baseline was evaluated using the harmonized model-visible geometry fields, its accuracy decreased to 0.6040 and its ROC-AUC to 0.4932. This result indicates that the preprocessing procedure substantially reduced the measured geometric separation between the cohorts. It does not establish that all scanner-, institution-, anatomy-, or cohort-specific signals were eliminated.

### 5.2.2 Training Behavior of the Individual Models

Training ended after six recorded epochs for ResNet50, seven for EfficientNetV2S, and 11 for DenseNet121 because validation-based early stopping was used. ResNet50 training loss decreased from 0.2612 to 0.0869, while training slice accuracy increased from 0.8878 to 0.9667. EfficientNetV2S training loss decreased from 0.3777 to 0.1843, with slice accuracy increasing from 0.8501 to 0.9274. DenseNet121 training loss decreased from 0.3934 to 0.1786, while slice accuracy increased from 0.8236 to 0.9281.

Patient-level validation ROC-AUC reached 1.0000 during the first recorded epoch for ResNet50 and the second recorded epoch for EfficientNetV2S. DenseNet121 reached its selected validation ROC-AUC of 0.9915 later in training. The early appearance of near-perfect validation separation suggests that the frozen ImageNet features and trained classification heads could readily distinguish the assigned cohorts. However, this behavior cannot by itself establish that the learned separation was specific to endometriosis. The strong association between class label and data source provides an alternative explanation that must be considered when interpreting both training and evaluation performance.

### 5.2.3 Patient-Level Validation Performance

Table 10 presents the patient-level validation estimates. EfficientNetV2S classified all 22 validation records according to their operational labels. ResNet50, DenseNet121, and the ensemble each classified 21 of 22 records correctly. Validation performance was used for checkpoint selection and training control; consequently, it represents model-development performance rather than an independent estimate of generalization.

**Table 10**  
*Patient-Level Validation Performance at the Fixed 0.5 Threshold*

| Model | ROC-AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1-score | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 1.0000 | 0.9545 | 0.9444 | 1.0000 | 0.8889 | 0.9286 | 0.9630 | 0.0451 |
| EfficientNetV2S | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0597 |
| DenseNet121 | 0.9915 | 0.9545 | 0.9444 | 1.0000 | 0.8889 | 0.9286 | 0.9630 | 0.0478 |
| Soft-voting ensemble | 1.0000 | 0.9545 | 0.9444 | 1.0000 | 0.8889 | 0.9286 | 0.9630 | 0.0447 |

Although EfficientNetV2S produced perfect threshold-based validation classifications, its Brier score was higher than those of the other models. Because the Brier score evaluates the squared difference between predicted probabilities and operational outcomes, perfect classification at a fixed threshold did not imply the best probabilistic accuracy.

### 5.2.4 Patient-Level Internal-Test Performance

Table 11 reports the locked internal-test results. The soft-voting ensemble achieved a ROC-AUC of 0.9841, accuracy of 0.9565, balanced accuracy of 0.9444, sensitivity of 1.0000, specificity of 0.8889, precision of 0.9333, F1-score of 0.9655, and Brier score of 0.0444. At the fixed threshold, the ensemble correctly classified all 14 UT-EndoMRI records and eight of the nine MOGaMBO proxy-comparator records. One proxy-comparator record received an operational-positive prediction. Because that record did not have a clinically verified endometriosis-negative reference standard, this outcome should be described as an operational cohort-classification error rather than a clinically established false-positive diagnosis.

**Table 11**  
*Patient-Level Internal-Test Performance at the Fixed 0.5 Threshold*

| Model | ROC-AUC | Accuracy | Balanced accuracy | Sensitivity | Specificity | Precision | F1-score | Brier score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet50 | 0.9841 | 0.9565 | 0.9444 | 1.0000 | 0.8889 | 0.9333 | 0.9655 | 0.0375 |
| EfficientNetV2S | 0.9762 | 0.9130 | 0.9087 | 0.9286 | 0.8889 | 0.9286 | 0.9286 | 0.0689 |
| DenseNet121 | 0.9841 | 0.9130 | 0.9087 | 0.9286 | 0.8889 | 0.9286 | 0.9286 | 0.0437 |
| Soft-voting ensemble | 0.9841 | 0.9565 | 0.9444 | 1.0000 | 0.8889 | 0.9333 | 0.9655 | 0.0444 |

Figure 6 displays the patient-level ROC curves. ResNet50, DenseNet121, and the ensemble had the same ROC-AUC of 0.9841, while EfficientNetV2S had a ROC-AUC of 0.9762. All four curves showed strong ranking separation on the internal test set. The small number of test records means that each point on the empirical curve represents a substantial fraction of the corresponding class, so minor visual differences should not be interpreted as evidence of meaningful superiority.

**Figure 6**  
*Internal-Test Patient-Level Receiver Operating Characteristic Curves*

> Insert `outputs/reporting/final/figures/test_roc_curves.pdf` here.

Figure 7 presents the model-specific confusion matrices. ResNet50 and the ensemble produced identical decisions: eight proxy-comparator records classified as proxy, one classified as operational-positive, and all 14 operational-positive records classified as positive. EfficientNetV2S and DenseNet121 each assigned a proxy prediction to one additional operational-positive record; these errors occurred on different records.

**Figure 7**  
*Internal-Test Patient-Level Confusion Matrices at the Fixed 0.5 Threshold*

> Insert `outputs/reporting/final/figures/test_confusion_matrices.pdf` here.

The 95% intervals in Figure 8 place the point estimates in the context of sampling uncertainty. For the ensemble, ROC-AUC was 0.9841 with a bootstrap 95% interval of 0.9286 to 1.0000. Accuracy was 0.9565 with a Wilson interval of 0.7901 to 0.9923. Sensitivity was 1.0000, but its Wilson interval extended from 0.7847 to 1.0000 because only 14 operational-positive records were available. Specificity was 0.8889 with an interval of 0.5650 to 0.9801 because it was estimated from only nine proxy-comparator records. Precision was 0.9333 with an interval of 0.7018 to 0.9881, and F1-score was 0.9655 with a bootstrap interval of 0.9032 to 1.0000. These intervals show that the high point estimates remain compatible with materially lower performance in another sample from the same underlying source populations.

**Figure 8**  
*Internal-Test Metrics With Primary 95% Confidence Intervals*

> Insert `outputs/reporting/final/figures/test_metrics_forest.pdf` here.

The ensemble did not uniformly outperform the individual networks. It tied ResNet50 on ROC-AUC and every threshold-based metric, and it tied DenseNet121 on ROC-AUC. ResNet50 also had the lowest test Brier score, 0.0375, compared with 0.0444 for the ensemble. EfficientNetV2S had produced the strongest threshold-based validation result but did not retain that advantage on the internal test set. Therefore, the evidence supports describing the ensemble as a stable aggregation of the three models, not as a statistically or uniformly superior classifier.

### 5.2.5 Model Agreement and Voting Patterns

The three base models produced unanimous binary decisions for 21 of the 23 internal-test records. The remaining two records produced two-to-one voting patterns, and the soft-voting decision did not differ from hard majority voting for any record. Pairwise overall agreement among the base models ranged from 0.9130 to 0.9565. Cohen's kappa was 0.9069 for ResNet50 versus EfficientNetV2S, 0.9069 for ResNet50 versus DenseNet121, and 0.8175 for EfficientNetV2S versus DenseNet121. Pearson correlations between base-model probabilities ranged from 0.9398 to 0.9733.

Figure 9 visualizes the pairwise kappa values. ResNet50 and the ensemble had a kappa of 1.000 because their threshold-based decisions were identical. EfficientNetV2S and DenseNet121 had the lowest pairwise kappa, although their agreement remained high. This pattern indicates that the three architectures produced somewhat different probability estimates but relatively little diversity in final binary decisions.

**Figure 9**  
*Internal-Test Pairwise Prediction Agreement Based on Cohen's Kappa*

> Insert `outputs/reporting/final/figures/model_agreement_matrix.pdf` here.

### 5.2.6 Grad-CAM Findings and Qualitative Themes

Grad-CAM was generated separately for ResNet50, EfficientNetV2S, and DenseNet121 on one representative slice from each of the 23 internal-test records, producing 69 model-specific heatmaps. The representative slice was selected as the slice whose ensemble probability was closest to that record's mean ensemble probability. Each heatmap therefore explained one component model's selected-slice prediction. It did not explain every slice that contributed to the patient-level probability and did not represent a single ensemble-level explanation.

Table 12 summarizes the exploratory artifact-attention measurements. Activation-mass fraction describes the proportion of total positive Grad-CAM activation located inside a defined region. Enrichment adjusts that fraction for the region's area: a value of 1.0 indicates activation proportional to region area, a value above 1.0 indicates relative concentration, and a value below 1.0 indicates relative underrepresentation.

**Table 12**  
*Median Exploratory Grad-CAM Artifact-Attention Measures Across 23 Representative Slices*

| Measure | ResNet50 | EfficientNetV2S | DenseNet121 |
|---|---:|---:|---:|
| 32-pixel border activation-mass fraction | 0.195 | 0.274 | 0.355 |
| 32-pixel border activation enrichment | 0.398 | 0.560 | 0.726 |
| Edge-connected background activation-mass fraction | 0.087 | 0.115 | 0.153 |
| Edge-connected background activation enrichment* | 0.353 | 0.566 | 0.665 |
| Foreground-intensity activation-mass fraction | 0.900 | 0.853 | 0.815 |
| Foreground-intensity activation enrichment | 1.267 | 1.211 | 1.169 |
| Heatmap-to-foreground center distance, pixels | 12.1 | 23.4 | 10.7 |

*Note.* Edge-connected background enrichment was available for 21 slices per model because the proxy region was absent in two selected slices. All measures are based on computational image regions rather than anatomical or lesion annotations.

Three qualitative themes emerged. First, median foreground enrichment exceeded 1.0 for all models, whereas median border and edge-connected background enrichment remained below 1.0; the typical heatmap was therefore not dominated by the border or zero-valued background. Second, patterns differed across architectures even when predicted classes agreed. EfficientNetV2S had the largest median heatmap-to-foreground center distance. Third, case-level variability remained, with several maps showing peripheral activation or border/background enrichment above 1.0, particularly for DenseNet121. Shortcut attention could not be excluded for individual cases.

Figure 10 displays the distribution of activation enrichment for the border, edge-connected background proxy, and foreground-intensity proxy. The figure should be interpreted as an artifact-attention audit rather than evidence that the models localized endometriosis lesions. The masks were not anatomical segmentations, the maps were enlarged from native 7 x 7 activations, and no radiologist or lesion annotation was available to determine whether highlighted areas were clinically relevant.

**Figure 10**  
*Exploratory Grad-CAM Artifact-Attention Audit Across 23 Representative Slices*

> Insert `outputs/reporting/final/figures/gradcam_artifact_attention.pdf` here.

## 5.3 Analysis and Interpretation

### 5.3.1 Internal Discrimination and Relation to Prior MRI Research

The internal results demonstrate that frozen-backbone transfer learning can produce strong separation between UT-EndoMRI and MOGaMBO records after patient-level probability aggregation. This observation is broadly consistent with previous evidence that ImageNet-pretrained CNNs can extract discriminative representations from MRI data despite limited medical-imaging sample sizes. Polat and Gungen (2021), for example, reported high brain-tumor MRI classification performance using pretrained ResNet50 and DenseNet121, while Noreen et al. (2020) showed that pretrained DenseNet-based features could support MRI classification through feature concatenation. These studies support the technical feasibility of transfer learning for MRI but do not validate the present model for endometriosis, because the anatomical target, class definitions, data sources, and reference standards differ.

The findings also relate to a small body of AI work involving endometriosis imaging. Liang et al. (2025) introduced UT-EndoMRI for pelvic-organ segmentation, Figueredo et al. (2024) segmented rectosigmoid deep endometriosis, and Peluso et al. (2026) investigated MRI-derived lesion features and histology. In contrast, this experiment performed patient-level cohort discrimination without lesion annotations, so its metrics are not directly comparable with segmentation or lesion-level results. Zhang et al. (2026) likewise found promising machine-learning performance alongside methodological limitations. Such caution is necessary here because discrimination cannot be separated from source-cohort differences.

### 5.3.2 Why the Ensemble Produced Limited Incremental Benefit

The theoretical benefit of ensemble learning depends not only on the accuracy of individual members but also on diversity in their errors (Dietterich, 2000). If component models make different mistakes, aggregation can reduce variance and recover correct decisions. In the present experiment, however, the base models agreed on 21 of 23 test records and their probability estimates were strongly correlated. All three models assigned the operational-positive prediction to the same proxy-comparator record. Equal averaging cannot correct an error shared by every component.

The two non-unanimous operational-positive records illustrate the limited but potentially useful role of aggregation. EfficientNetV2S classified one of these records below the threshold, while DenseNet121 classified the other below the threshold; ResNet50 classified both as positive. Averaging retained the positive ensemble decision for each record. This improved the ensemble's threshold-based performance relative to EfficientNetV2S and DenseNet121, but it did not improve upon ResNet50. The ensemble therefore provided protection against isolated component errors, yet the observed advantage arose from only two records and should not be generalized as evidence of consistent ensemble superiority.

The Brier scores further qualify the result. ResNet50's lower score indicates that averaging did not improve probabilistic accuracy in this sample. Equal weighting was prespecified, no calibration or weight optimization was performed, and highly correlated models offer less opportunity for improvement. Future ensemble development should therefore evaluate complementarity and calibration rather than assuming that architectural variety guarantees diverse errors.

### 5.3.3 Sampling Uncertainty and the Meaning of Perfect Sensitivity

The ensemble identified all 14 operational-positive test records, producing a sensitivity point estimate of 1.0000. This result is encouraging for the frozen internal split, but it should not be described as perfect or certain detection. The Wilson interval extended down to 0.7847, showing the uncertainty associated with only 14 positive records. Specificity was based on nine proxy-comparator records and had an even wider interval of 0.5650 to 0.9801. A single additional error would materially alter these estimates.

The near-perfect AUC values indicate strong ranking on the observed records, but the small test set cannot represent the full variation in scanners, anatomy, disease phenotype, comorbidity, and acquisition encountered clinically. The confidence intervals therefore provide a more responsible account than the point estimates alone.

### 5.3.4 Source-Cohort Confounding and Potential Shortcut Learning

The main limitation of the internal experiment is that operational label and source cohort were completely confounded. Every operational-positive record originated from UT-EndoMRI, whereas every proxy-comparator record originated from MOGaMBO. The cohorts also differed clinically: one was assembled around endometriosis and the other around cervical-cancer brachytherapy imaging. The model could therefore have used a mixture of disease-related anatomy, other pathology, scanner or protocol characteristics, institutional conventions, field-of-view differences, or residual preprocessing cues.

The geometry baseline provides evidence that harmonization reduced a major class of shortcuts. Original dimensions and voxel spacing nearly separated the cohorts by themselves, while the corresponding harmonized model-visible geometry performed near chance by ROC-AUC. Nevertheless, this check addressed only measured geometry fields. It did not test all intensity, texture, anatomy, pathology, or acquisition differences. Zech et al. (2018) demonstrated that a high-performing radiographic classifier could show variable generalization across hospitals because institutional and prevalence differences affected learned behavior. DeGrave et al. (2021) similarly found that radiographic deep-learning systems could rely on shortcuts rather than intended disease signals. These studies align with the present concern that strong internal performance can coexist with uncertain disease specificity.

The results therefore support internal cohort discrimination only. They do not establish discrimination from healthy pelvis, other gynecological disorders, or clinically verified endometriosis-negative patients. An independently sourced hospital cohort with verified sequences and reference labels is needed to test performance beyond the two development sources. The intended population, reference standard, and inclusion criteria must also be specified before diagnostic claims are made.

### 5.3.5 Interpretation and Limits of Grad-CAM

Grad-CAM was included to expose image regions that influenced each component CNN's selected-slice prediction, consistent with the method introduced by Selvaraju et al. (2017). The foreground-enrichment findings offer some reassurance that median activation was not disproportionately concentrated in borders or zero-valued padding. At the same time, these results do not reveal whether the models attended to endometriosis lesions. The foreground proxy included all sufficiently intense anatomy, not disease-specific structures, and individual heatmaps displayed considerable variation.

This distinction is consistent with broader XAI literature. Ali et al. (2023) and Muhammad and Bendechache (2024) emphasized that explainability methods can improve transparency but require careful validation. Ghassemi et al. (2021) further argued that current post hoc explanations in health care may create false reassurance when they are treated as evidence of causal or clinically meaningful reasoning. In the present study, a visually plausible heatmap cannot prove that the model used a medically appropriate feature. Conversely, peripheral activation does not prove that the prediction was entirely based on an artifact.

Additional constraints apply to the generated maps. Each explanation represented one deterministic slice rather than the full set of slices used in patient-level mean pooling. The three component networks had separate Grad-CAM maps, and the ensemble had no shared convolutional feature map from which to derive a single ensemble explanation. The native maps were only 7 x 7 before cubic enlargement to 224 x 224, limiting spatial precision. Finally, the transferred Grad-CAM provenance did not include a producer-side heatmap hash manifest. The reporting audit recorded the hashes of the received maps for future integrity checks, but it could not cryptographically prove that they were unchanged from the original generation run. These limitations justify presenting the heatmaps as exploratory model-attribution outputs pending radiologist review.

### 5.3.6 Implications, Reporting Boundaries, and Future Validation

The completed experiment achieved its computational objective of constructing and internally evaluating an equal-weight soft-voting framework using three frozen ImageNet backbones. It also demonstrated a reproducible patient-level evaluation pipeline, exact-volume deduplication, source-stratified patient partitioning, confidence-interval reporting, agreement analysis, and deterministic Grad-CAM generation. These are methodological strengths because they reduce slice-level leakage and make the origin of reported results auditable. The reporting approach is consistent with the transparency goals of the Checklist for Artificial Intelligence in Medical Imaging (Mongan et al., 2020) and the updated TRIPOD+AI guidance for prediction-model reporting (Collins et al., 2024).

The findings do not yet support deployment as a clinical decision-support tool. External hospital testing has not been completed, expert review of the Grad-CAM maps remains pending, and the internal comparator does not represent the intended clinical differential diagnosis. The study also used only T2-weighted images, frozen feature extractors, and a small held-out test set. No lesion-level ground truth, patient symptoms, disease stage, treatment history, or consistent comorbidity data were available. In addition, no formal paired comparison demonstrated that the ensemble was statistically superior to its component models.

Future work should prioritize validation rather than further optimization on the internal split. Hospital data should be evaluated only after ethical determination, T2 sequence verification, and radiologist-supported labels. A stronger design would include confirmed positive and clinically appropriate negative or differential-diagnosis cases from the same target setting. Calibration, failure analysis, and confidence intervals should accompany discrimination results. Grad-CAM should be assessed against expert annotations and across slices when possible.

## 5.4 Chapter Summary

The soft-voting ensemble achieved high patient-level discrimination on the locked internal test set, with a ROC-AUC of 0.9841 and an accuracy of 0.9565. It classified all 14 UT-EndoMRI operational-positive records according to their cohort label and eight of nine MOGaMBO proxy-comparator records. However, the ensemble did not outperform ResNet50 on the internal test metrics and did not achieve the lowest Brier score. The high agreement and probability correlation among the component models explain why equal-weight averaging provided only limited incremental benefit.

The Grad-CAM audit found that median activation was relatively concentrated in the foreground-intensity proxy rather than disproportionately concentrated in border or zero-background regions, although individual maps remained heterogeneous and sometimes peripheral. Because the maps explained only one slice per component model and lacked lesion annotations and expert review, they cannot be interpreted as validated lesion localization.

Overall, the results establish technical feasibility and strong internal cohort discrimination under the frozen experimental protocol. They do not establish clinical diagnostic validity. The complete association between source cohort and operational label, the proxy nature of the comparator, the small internal test set, and the absence of external and expert validation remain the principal constraints on interpretation.

---

## Bibliography Entries to Add

The following verified sources are cited in this chapter but do not appear in the supplied bibliography:

Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B., Ghassemi, M., Liu, X., Reitsma, J. B., van Smeden, M., Boulesteix, A.-L., Camaradou, J. C., Celi, L. A., Denaxas, S., Denniston, A. K., Glocker, B., Golub, R. M., Harvey, H., Heinze, G., . . . Logullo, P. (2024). TRIPOD+AI statement: Updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ, 385*, e078378. https://doi.org/10.1136/bmj-2023-078378

DeGrave, A. J., Janizek, J. D., & Lee, S.-I. (2021). AI for radiographic COVID-19 detection selects shortcuts over signal. *Nature Machine Intelligence, 3*(7), 610-619. https://doi.org/10.1038/s42256-021-00338-7

Ghassemi, M., Oakden-Rayner, L., & Beam, A. L. (2021). The false hope of current approaches to explainable artificial intelligence in health care. *The Lancet Digital Health, 3*(11), e745-e750. https://doi.org/10.1016/S2589-7500(21)00208-9

Mongan, J., Moy, L., & Kahn, C. E., Jr. (2020). Checklist for Artificial Intelligence in Medical Imaging (CLAIM): A guide for authors and reviewers. *Radiology: Artificial Intelligence, 2*(2), e200029. https://doi.org/10.1148/ryai.2020200029

Zech, J. R., Badgeley, M. A., Liu, M., Costa, A. B., Titano, J. J., & Oermann, E. K. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study. *PLOS Medicine, 15*(11), e1002683. https://doi.org/10.1371/journal.pmed.1002683

---

## Mandatory Consistency Corrections Outside Chapter 5

These corrections are not part of the Chapter 5 prose, but they should be made before integrating the chapter into the thesis:

1. Replace `EfficientNetV2` with `EfficientNetV2S` throughout the abstract, objectives, technical background, and Chapter 5.
2. Remove the abstract claim that hospital data underwent training or evaluation. The hospital dataset remained excluded from the completed experiment.
3. Replace anticipated claims of ensemble superiority with the actual finding that the ensemble tied ResNet50 on internal-test ROC-AUC and threshold-based metrics.
4. Remove claims that Grad-CAM highlighted clinically relevant anatomy or improved clinical confidence. Expert review and lesion-localization validation remain pending.
5. Remove the claim that the framework is a reliable clinical decision-support tool. The current evidence supports technical feasibility and internal cohort discrimination only.
6. Correct Table 1's MOGaMBO classification label from `Healthy Pelvis` to `Cervical-cancer proxy comparator; endometriosis-negative status not confirmed`.
7. Correct Table 1's raw dataset counts and distinguish raw source folders from the 149 deduplicated retained records.
8. Replace Table 2's output description `presence or absence of endometriosis` with `operational UT-EndoMRI versus MOGaMBO cohort label`.
9. Delete the obsolete second `Data Partitioning and Augmentation` subsection on manuscript pages 34-35, which still describes an 80:20 public-data split and the hospital dataset as the test set.
10. Remove future-tense statements that external testing was completed or formed part of final testing. Describe it as pending.
11. Update the automated-test count in Section 4.10.1 if it is retained; the repository now passes 40 tests, not 28.
12. Replace the statement that validation was used to establish interpretability through clinically relevant structures. Grad-CAM was technically verified and explored, not clinically validated.
13. Update the abstract, scope, significance, and conceptual framework so they consistently describe source-confounded internal cohort discrimination rather than validated endometriosis diagnosis.
14. Update the list of figures and list of tables to include Tables 9-12 and Figures 6-10 after final pagination.
