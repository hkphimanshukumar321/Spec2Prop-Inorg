# Spec2Prop-Inorg Dataset

## Project Objective
The objective of this project is to provide a lightweight, open-data CNN framework for spectra-to-property screening of inorganic compounds. This derived dataset contains aligned Raman and XRD spectra with computed material properties for deep learning tasks.

## Source Datasets
This is a derived/processed benchmark dataset created from open scientific datasets. It does not claim ownership over the original source datasets. The underlying datasets include:
- **RRUFF Database**: The main Raman/XRD mineral training backbone. Provides open Raman and XRD spectra for minerals.
- **Matbench**: Provides computed material properties (band gap, formation energy, metallicity) from the Materials Project.
- **MLROD**: An auxiliary Raman resource.
- **tmQM / tmQMg**: Optional coordination chemistry resources containing organometallic and transition metal complexes, not the main training backbone.

*Note on COD (Crystallography Open Database)*: COD was considered but deferred because it mainly provides CIF crystal structures, not directly paired Raman spectra. COD may be used in future v2 for simulated XRD or structural descriptors.

## Created Subsets
The preprocessing pipeline produces the following subsets to accommodate different experimental setups:
- **Full Spectra Subset (`spec2prop_full_spectra`)**: Contains all processed RRUFF spectra, including unknown, organic, and mixed samples.
- **Clean Inorganic Subset (`spec2prop_clean_inorganic`)**: Filters out `Unknown`, `Mixed/Other`, and `Organic` families, leaving only clean inorganic samples.
- **Property Matched Inorganic Subset (`spec2prop_property_matched_inorganic`)**: A subset of the clean inorganic data where the compound was successfully matched with computational properties in Matbench.
- **XRD Linked Inorganic Subset (`spec2prop_xrd_linked_inorganic`)**: A subset of the clean inorganic data that has paired X-ray diffraction (XRD) data available.

## Task Definitions
The dataset is primarily designed for the following predictive tasks using convolutional neural networks (CNNs):
- Regression of band gap (eV) and formation energy (eV/atom) from Raman/XRD spectra.
- Classification of `is_metal` and `band_gap_class`.
- Multi-modal (dual-branch) representation learning combining Raman and XRD inputs.

## Pipeline Architecture
![Spec2Prop Pipeline Architecture](C:/Users/hkphi/.gemini/antigravity-ide/brain/28d63add-1802-4819-abe6-0c17a67db245/spec2prop_pipeline_diagram_1782472059505.png)

The workflow consists of five major stages bridging raw scientific datasets to deep learning architectures:

1. **Open-Access Input Data Integration**: 
   - We ingest unaligned raw spectral files from the **RRUFF database** (Raman and XRD) alongside computational property databases like **Matbench (Materials Project)** and **tmQM/tmQMg**.

2. **Preprocessing and Benchmark Construction**: 
   - **Spectral Preprocessing**: Raw Raman shifts (100–4000 cm⁻¹) and XRD 2-theta angles (5°–90°) are cleaned, baseline-corrected, and interpolated to a uniform 2048-point fixed vector grid. Max normalization is applied to standardize intensities.
   - **Metadata Extraction & Formula Normalization**: Complex string parsing is used to pull chemical formulas, which are then stripped of noise (e.g., HTML, structural markers) and reduced to an empirical format for cross-database joining.
   - **Chemical Annotation**: Utilizing elemental compositions, the compounds are programmatically assigned into a 15-tier hierarchy of chemical families (e.g., Silicates, Oxides, Sulfides).
   - **Dataset Matching & Fusion**: Cleaned empirical formulas serve as a primary key to reliably pull computational targets (band gap, formation energy) from Matbench into the spectral metadata. Deterministic many-to-one merging strategies ensure zero row explosion.

3. **Final Processed Dataset Subsets**: 
   - The preprocessing outputs specialized data tiers (`Full Spectra`, `Clean Inorganic`, `Property-Matched`, and `XRD-Linked`) bundled with data-leakage safe splits ensuring reliable cross-validation.

4. **AI / ML Framework**: 
   - The designed architecture takes advantage of a multi-branch neural network. 1D-CNN encoders extract features from Raman and XRD spectral fingerprints. These are concatenated with a multi-layer perceptron (MLP) branch processing chemical descriptor priors.
   
5. **Learning Tasks and Outputs**: 
   - A multi-task learning head allows for:
      - **Task 1**: Broad chemical family classification (silicate vs oxide vs sulfate).
      - **Task 2**: Direct spectra-to-property prediction (regressing formation energy, classifying metallicity).
      - **Task 3**: Multi-modal fusion where combined Raman + XRD input yields superior material prediction.

## Formula Cleanup Rules
Raw formulas from RRUFF are normalized, stripped of formatting noise (e.g., HTML, specific RRUFF artifacts), and parsed using `pymatgen`. Variables (`n`), mixed occupancies, vacancies (`[box]`), and hydrates are flagged to avoid polluting property prediction mappings.

## Chemical Family Rules
Using the chemical formula and elemental composition, minerals are classified into a 15-tier hierarchy including:
Elemental/Native, Silicate (with Silica sub-family), Carbonate, Sulfate, Phosphate, Borate, Halide, Sulfide, Oxide, Carbide, Telluride/Selenide, Intermetallic/Alloy.
- **Silica Sub-family**: Common silica polymorphs (Quartz, Tridymite, Cristobalite, Coesite, Opal, Moganite, Chalcedony) are assigned a `Silica` sub-family label.
- **Organic Exclusion**: Any obvious organic molecules or compounds strictly containing C, H, O (without metals) are flagged via `exclude_from_inorganic_main`.

## Matbench Matching Strategy
Formulas that parse successfully and cleanly (no vacancies, variables, or solid solutions) are reduced to their empirical formula. This `reduced_formula` is used as a join key against the Matbench targets (`matbench_mp_gap`, `matbench_mp_e_form`, `matbench_mp_is_metal`). Since multiple computational structures may exist for the same formula, the median values are taken.

## XRD Linking Strategy
The dataset groups X-ray diffraction patterns by `rruff_id` to establish paired Raman-XRD samples.

## Leakage-Safe Splitting Strategy
Splits (train/val/test: 70/15/15) are provided to ensure no data leakage occurs across sets.
- **Grouping**: Samples with the same `rruff_id` are forced into the same split. If `rruff_id` is missing, `reduced_formula` is used, followed by `mineral_name`.
- **Stratification**: Splits aim to stratify across `chemical_family` and `band_gap_class` to ensure proportional representation.

## Limitations
- **Theoretical vs Experimental**: Properties provided (band gap, formation energy) are from DFT calculations (Materials Project) rather than experimental measurements.
- **Phase Ambiguity**: A single chemical formula might map to multiple crystal structures (polymorphs). The property values represent aggregated computational predictions for that formula, which may differ from the specific polymorph's actual experimental values.
- **Measurement Variance**: RRUFF Raman spectra are taken under different lasers and orientations; this variation is preserved as data augmentation, but flagged with `duplicate_rruff_id`.

## Citation Requirements
If you use this dataset, please cite the Spec2Prop-Inorg framework as well as the underlying source datasets: RRUFF Database, Materials Project (Matbench), and tmQM/tmQMg.
