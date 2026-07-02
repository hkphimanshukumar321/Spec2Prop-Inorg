You are an expert scientific manuscript writer, LaTeX editor, and materials-informatics researcher. I want you to help me frame a full journal manuscript for **Chemistry Africa / Springer Nature format** in LaTeX.

The paper topic is:

**Spec2Prop-InorgBench: An Open Raman–XRD Benchmark and AI-Assisted Screening Workflow for Inorganic Materials**

This paper should be **chemistry-heavy in motivation, dataset construction, spectral interpretation, family definitions, and discussion**, while the **Materials and Methods section should be ML-heavy with proper model details, evaluation protocol, preprocessing, feature extraction, and leakage-safe validation**.

Important: Do not make the paper look like a generic ML classification paper. It should look like a chemistry/materials-informatics paper focused on Raman/XRD spectral screening of inorganic materials.

---

## 1. Main contribution and story

Frame the paper around these contributions:

1. A curated open-data inorganic Raman–XRD benchmark named **Spec2Prop-InorgBench**.
2. Runtime preprocessing pipeline for raw Raman and optional XRD files.
3. Chemistry-aware spectral feature extraction using PCA and expert descriptors.
4. AI-assisted inorganic material family screening using LightGBM-DART.
5. Spectra-to-property prediction for band-gap class, metal/non-metal class, and formation-energy class.
6. Dataset-backed deployment demonstration named **Spec2Prop-Edge**.
7. Confidence-aware screening assistant, not final experimental material identification.

Use the following exact dataset statistics:

* Full usable Raman spectra: **4,118**
* Clean inorganic samples: **4,027**
* Property-matched samples: **1,844**
* Raman–XRD linked samples: **1,462**
* Raman vector size: **2048 points**
* Final compact feature vector: **158 dimensions**

  * **32 PCA global-shape features**
  * **126 expert spectral descriptors**
* Main classifier: **LightGBM-DART**
* Chemical family classification result:

  * Accuracy: **67.22%**
  * Macro-F1: **56.57%**
* Spectra-to-property results:

  * Band-gap class macro-F1: **61.09%**
  * Metal/non-metal macro-F1: **67.56%**
  * Formation-energy class macro-F1: **78.91%**
* Classes/families:

  * Silicate
  * Oxide
  * Carbonate
  * Sulfate
  * Phosphate
  * Sulfide
  * Halide
  * Borate
  * Other/Rare

Dataset DOI to include in the non-blinded version:

**H. Kumar and K. Amrita, Spec2Prop-InorgBench, Kaggle, 2026, doi: 10.34740/KAGGLE/DSV/17271313.**

But because Chemistry Africa uses double-blind review, create two versions:

1. **blinded_main.tex**: no author names, no Kaggle author-identifying link, no personal GitHub/demo URL.
2. **main_nonblind.tex**: includes authors, dataset DOI, demo URL, acknowledgements, and code/data availability.

---

## 2. Required LaTeX project structure

Create a complete LaTeX project using subfiles. Use Springer Nature compatible LaTeX style as much as possible. Make the project modular.

Generate this folder/file structure:

```text
Spec2Prop_ChemistryAfrica/
├── main_blinded.tex
├── main_nonblind.tex
├── references.bib
├── sections/
│   ├── 00_titlepage_nonblind.tex
│   ├── 01_abstract_keywords.tex
│   ├── 02_introduction.tex
│   ├── 03_materials_methods.tex
│   ├── 04_results_discussion.tex
│   ├── 05_limitations.tex
│   ├── 06_conclusion.tex
│   ├── 07_declarations_blinded.tex
│   ├── 08_declarations_nonblind.tex
├── tables/
│   ├── table1_data_sources.tex
│   ├── table2_dataset_subsets.tex
│   ├── table3_family_definitions.tex
│   ├── table4_preprocessing_operations.tex
│   ├── table5_feature_groups.tex
│   ├── table6_model_configurations.tex
│   ├── table7_evaluation_protocol.tex
│   ├── table8_family_classification_results.tex
│   ├── table9_property_prediction_results.tex
│   ├── table10_limitations_future_work.tex
├── figures/
│   ├── fig1_overall_workflow.pdf
│   ├── fig2_dataset_funnel.pdf
│   ├── fig3_runtime_preprocessing.pdf
│   ├── fig4_feature_model_architecture.pdf
│   ├── fig5_family_distribution.pdf
│   ├── fig6_confusion_matrix.pdf
│   ├── fig7_feature_importance_shap.pdf
│   ├── fig8_property_prediction.pdf
│   ├── fig9_deployment_demo.pdf
├── supplementary/
│   ├── supplementary.tex
│   ├── supp_tables.tex
│   ├── supp_figures.tex
└── README_compile.md
```

Use `\input{}` for all section and table files. Add figure placeholders using `\includegraphics` and `\fbox` placeholders if actual images are not available.

For every table/figure placeholder, include a caption and a clear TODO comment telling what data/image should be inserted.

---

## 3. Paper title and author information

Use this main title:

**Spec2Prop-InorgBench: An Open Raman–XRD Benchmark and AI-Assisted Screening Workflow for Inorganic Materials**

Alternative subtitle inside abstract/introduction:

**A confidence-aware Raman/XRD spectral screening framework for inorganic material family and property prediction**

Non-blind author details:

* **Kumari Amrita**

  * Department of Chemistry, National Institute of Technology Kurukshetra, Kurukshetra, Haryana, India
  * Email: [725101023@nitkkr.ac.in](mailto:725101023@nitkkr.ac.in)

* **Himanshu Kumar**

  * School of Electrical and Computer Sciences, Indian Institute of Technology Bhubaneswar, Bhubaneswar, Odisha, India
  * Email: [24sp06015@iitbbs.ac.in](mailto:24sp06015@iitbbs.ac.in)

Corresponding author: Himanshu Kumar, unless otherwise required.

---

## 4. Required manuscript section structure

Use this exact structure:

```latex
\begin{abstract}
...
\end{abstract}

\keywords{Inorganic materials \and Raman spectroscopy \and X-ray diffraction \and Materials informatics \and Spectral preprocessing \and LightGBM \and Spectra-to-property prediction}

\section{Introduction}

\section{Materials and Methods}
\subsection{Open spectral and materials-property resources}
\subsection{Construction of Spec2Prop-InorgBench}
\subsection{Inorganic chemical-family label construction}
\subsection{Runtime Raman and XRD spectral preprocessing}
\subsection{Chemistry-aware spectral feature extraction}
\subsection{Machine-learning models}
\subsection{Leakage-safe evaluation protocol}
\subsection{Confidence-aware deployment workflow}

\section{Results and Discussion}
\subsection{Dataset composition and inorganic chemical-family distribution}
\subsection{Raman-based inorganic family screening}
\subsection{Chemical interpretation of spectral features}
\subsection{Spectra-to-property prediction}
\subsection{Raman--XRD multimodal analysis}
\subsection{Spec2Prop-Edge deployment demonstration}

\section{Limitations and Future Work}

\section{Conclusion}

\section*{Declarations}
```

---

## 5. Page-wise content plan

Frame the paper as approximately 12–15 manuscript pages.

### Page 1

Title, abstract, keywords.

Abstract must include:

* Chemistry problem
* Raman/XRD relevance
* Dataset gap
* Dataset numbers
* Preprocessing and feature extraction
* LightGBM-DART result
* Property prediction result
* Screening-not-final-confirmation statement

### Pages 2–3: Introduction

Make it chemistry-heavy.

Cover:

* Raman spectroscopy as vibrational fingerprinting for inorganic materials.
* XRD as structural/crystallographic fingerprinting.
* Importance of silicates, oxides, carbonates, sulfates, phosphates, sulfides, halides, and borates.
* Problems with raw spectral files: nonuniform axes, baseline drift, noise, instrument variation, scaling, duplicate points, invalid rows.
* Need for open reproducible Raman/XRD benchmark.
* Gap: many ML works classify spectra, but fewer provide full raw-file-to-screening workflow with dataset release, property prediction, leakage-safe evaluation, and deployment demo.

Insert **Figure 1** near the end of Introduction:

**Figure 1. Overall Spec2Prop-InorgBench workflow.**

Placeholder caption:
“Open Raman and XRD spectral resources are curated, formula-normalized, linked with inorganic family and materials-property labels, transformed into model-ready spectral representations, and used for confidence-aware screening.”

### Pages 4–8: Materials and Methods

This should be ML-heavy but chemically justified.

Include these tables:

* Table 1: Open data sources
* Table 2: Dataset subset statistics
* Table 3: Chemical family definitions
* Table 4: Preprocessing operations
* Table 5: Feature groups
* Table 6: Model configurations
* Table 7: Evaluation protocol

Insert **Figure 2**: dataset construction funnel.
Insert **Figure 3**: runtime spectral preprocessing.
Insert **Figure 4**: feature extraction + LightGBM-DART architecture.

### Pages 9–13: Results and Discussion

Make discussion chemistry-heavy.

Include:

* Chemical family distribution
* Performance comparison
* Confusion matrix
* Feature importance/SHAP with chemical interpretation
* Property prediction
* Raman–XRD multimodal extension
* Deployment demo

Insert:

* Figure 5: family distribution
* Figure 6: confusion matrix
* Figure 7: feature importance/SHAP
* Figure 8: property prediction results
* Figure 9: deployment demo screenshot, optional or supplementary

Main results table:

* Table 8: family classification results
* Table 9: property prediction results

### Pages 14–15

Limitations, future work, conclusion, declarations.

---

## 6. Table templates to generate

Generate real LaTeX tables with placeholders where values are not finalized. Use `TBD` instead of inventing numbers.

### Table 1: Open data sources

Columns:

* Source
* Data type
* Main fields used
* Linkage key
* Role in benchmark
* Status

Rows:

* RRUFF Raman
* RRUFF XRD
* RRUFF metadata/chemistry
* Materials Project/Matbench-derived labels
* Spec2Prop-InorgBench Kaggle release

### Table 2: Dataset subsets

Columns:

* Subset
* Samples
* Raman available
* XRD available
* Property labels
* Main use

Rows:

* Full spectra subset: 4,118
* Clean inorganic subset: 4,027
* Property-matched subset: 1,844
* Raman–XRD linked subset: 1,462

### Table 3: Chemical family definitions

Columns:

* Family
* Chemical meaning
* Representative spectral features
* Model role

Rows:

* Silicate
* Oxide
* Carbonate
* Sulfate
* Phosphate
* Sulfide
* Halide
* Borate
* Other/Rare

### Table 4: Preprocessing operations

Columns:

* Step
* Operation
* Chemistry/spectral reason
* ML reason
* Output

Rows:

* raw file parsing
* invalid row removal
* axis sorting
* duplicate removal
* interpolation to 2048 points
* baseline correction
* Savitzky–Golay smoothing
* min–max normalization

### Table 5: Feature groups

Columns:

* Feature group
* Dimension
* Chemical interpretation
* Used in final model

Rows:

* PCA global-shape features: 32
* Expert descriptors: 126
* Peak statistics: TBD
* Spectral moments: TBD
* Diagnostic band ratios: TBD
* Compact spectral-shape features: TBD
* Total vector: 158

### Table 6: Model configurations

Columns:

* Model
* Input representation
* Output task
* Purpose
* Key settings

Rows:

* Logistic regression
* SVM
* Random forest
* XGBoost
* LightGBM-DART
* SimpleCNN1D
* RamanFormer1D
* FusionSpec2PropNet
* DualBranch Raman–XRD model

### Table 7: Evaluation protocol

Columns:

* Task
* Dataset subset
* Split strategy
* Metrics
* Leakage control

Rows:

* family classification
* property prediction
* Raman–XRD multimodal classification
* confidence-aware screening

### Table 8: Family classification results

Columns:

* Model
* Input
* Accuracy
* Macro-F1
* Notes

Use exact final values where available:

* LightGBM-DART: accuracy 67.22, macro-F1 56.57

Other rows can be TBD:

* Logistic regression: TBD
* SVM: TBD
* Random forest: TBD
* XGBoost: TBD
* SimpleCNN1D: TBD
* RamanFormer1D: TBD

### Table 9: Property prediction results

Columns:

* Task
* Best model
* Accuracy
* Macro-F1
* Chemistry relevance

Use exact values:

* Band-gap class: macro-F1 61.09
* Metal/non-metal: macro-F1 67.56
* Formation-energy class: macro-F1 78.91

Accuracy can be TBD unless provided.

---

## 7. Figure placeholders to generate

For every figure, create a LaTeX placeholder with caption and TODO instruction.

### Figure 1

Overall Spec2Prop-InorgBench workflow.

### Figure 2

Dataset construction funnel:
Raw records → 4,118 usable Raman spectra → 4,027 clean inorganic samples → 1,844 property-matched samples → 1,462 Raman–XRD linked samples.

### Figure 3

Runtime Raman/XRD preprocessing:
Raw file parsing → invalid row removal → sorting/deduplication → interpolation → baseline correction → smoothing → normalization → 2048-point vector.

### Figure 4

Feature extraction and LightGBM-DART architecture:
2048-point Raman vector → 32 PCA features + 126 expert descriptors → 158-dimensional vector → LightGBM-DART → top-k chemical family + confidence score.

### Figure 5

Chemical family distribution bar chart.

### Figure 6

Normalized confusion matrix.

### Figure 7

SHAP or feature importance plot.

### Figure 8

Property prediction macro-F1 bar chart.

### Figure 9

Spec2Prop-Edge deployment demo screenshot.

---

## 8. Citation strategy

Add citations in every section and subsection. Do not leave any section without citations.

Use citations for:

* Raman spectroscopy and mineral/material identification
* XRD and powder diffraction analysis
* RRUFF database
* Materials Project
* Matbench
* Raman preprocessing
* baseline correction
* Savitzky–Golay smoothing
* spectral feature extraction
* PCA
* LightGBM / GBDT / DART
* XGBoost / random forest / SVM baselines
* neural spectral models
* Raman ML
* XRD ML
* materials informatics
* uncertainty-aware ML
* calibration/confidence scoring
* SHAP/explainable AI
* data leakage and model validation
* open scientific datasets and reproducibility

Important:

* Use **at least 40 references** in `references.bib`.
* Prioritize **latest 2023–2026 references** wherever possible.
* Foundational references are allowed only when necessary, for example:

  * Savitzky–Golay
  * PCA
  * Random Forest
  * SVM
  * XGBoost
  * LightGBM
  * SHAP
  * RRUFF
  * Materials Project
  * Matbench
* Do **not fabricate references**.
* If you cannot verify a reference title, DOI, volume, issue, or page range, mark it as `TBD_VERIFY` instead of inventing details.
* Create a separate file `refs_to_verify.md` listing any uncertain references.
* In `references.bib`, include DOI whenever known.
* Use clean BibTeX keys like:

  * `dai2024libsraman`
  * `rincon2025simpod`
  * `guo2024xrd`
  * `ke2017lightgbm`
  * `dunn2020matbench`
  * `lafuente2015rruff`

---

## 9. Required reference categories

Build the bibliography with at least 40 references from the following categories.

### Category A: Raman spectroscopy and mineral/inorganic material identification

Include recent papers from 2023–2026 on:

* Raman mineral identification
* Raman spectral classification
* Raman + machine learning
* Raman preprocessing
* Raman spectral databases
* LIBS–Raman or multimodal spectroscopy for minerals/materials

### Category B: XRD and machine learning

Include recent papers from 2023–2026 on:

* powder XRD benchmark datasets
* XRD pattern classification
* phase identification
* XRD to structure/property prediction
* deep learning for diffraction data

### Category C: Materials informatics and property prediction

Include:

* Materials Project
* Matbench
* materials-property prediction
* band-gap prediction
* formation energy prediction
* open materials databases
* inorganic materials screening

### Category D: ML methods used in this paper

Include:

* LightGBM
* DART/dropout boosting if available
* XGBoost
* Random Forest
* SVM
* PCA
* calibration/confidence scoring
* SHAP/explainable ML
* leakage-safe evaluation

### Category E: Spectral preprocessing

Include:

* Savitzky–Golay smoothing
* baseline correction
* normalization
* spectral interpolation
* chemometric preprocessing
* Raman baseline correction papers

---

## 10. Writing style

Write in polished journal style.

Avoid overclaiming. Use terms such as:

* “screening”
* “candidate prioritization”
* “decision support”
* “confidence-aware prediction”
* “preliminary material-family interpretation”
* “scientist verification”

Avoid terms such as:

* “final identification”
* “fully automatic discovery”
* “replaces laboratory analysis”
* “guaranteed classification”

Use this exact positioning statement in the Discussion or Conclusion:

“Spec2Prop-Edge is intended as a confidence-aware screening assistant for candidate prioritization and preliminary material-family interpretation. It does not replace expert spectroscopic interpretation, crystallographic validation, or laboratory confirmation.”

---

## 11. Required content for each subsection

### Introduction

Include:

* Raman/XRD complementarity
* importance of inorganic family screening
* open-data gap
* need for reproducible preprocessing
* limitations of raw spectral files
* why ML can help but must be confidence-aware
* contributions

### Materials and Methods

Include:

* data sources
* formula normalization
* family label construction
* property label matching
* Raman preprocessing
* XRD preprocessing
* feature extraction
* LightGBM-DART
* baselines
* leakage-safe split
* metrics
* deployment workflow

### Results and Discussion

Include:

* dataset composition
* chemical family distribution
* family classification results
* confusion matrix interpretation
* chemically meaningful feature importance
* property prediction results
* Raman–XRD multimodal analysis
* deployment demo
* limitations

### Conclusion

Include:

* benchmark contribution
* preprocessing contribution
* model contribution
* screening/demo contribution
* limitations and future validation

---

## 12. Expected output from you

First, produce the complete LaTeX project content in this order:

1. `main_blinded.tex`
2. `main_nonblind.tex`
3. each file in `sections/`
4. each file in `tables/`
5. `supplementary/supplementary.tex`
6. `references.bib`
7. `refs_to_verify.md`
8. `README_compile.md`

For each file, show the filename as a heading before the code block.

Use LaTeX code blocks.

Do not skip files.

Do not invent final numerical results where I have written TBD.

Do not fabricate references. Use verified references where possible and mark uncertain items clearly as `TBD_VERIFY`.

The final manuscript should be ready for me to paste into Overleaf and compile after I add actual figures.
