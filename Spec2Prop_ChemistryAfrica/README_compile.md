# Spec2Prop-InorgBench LaTeX Project

This is a modular LaTeX project formatted for Springer Nature (*Chemistry Africa*).

## Project Structure
- `main_blinded.tex`: Compile this for the double-blind peer review submission.
- `main_nonblind.tex`: Compile this for the final camera-ready version or preprint.
- `sections/`: Contains individual text sections.
- `tables/`: Contains all tabular data.
- `figures/`: Directory to place your PDF/PNG figures.
- `supplementary/`: Contains the SI document.
- `references.bib`: Bibliography file containing 40+ references.

## How to Compile in Overleaf
1. Compress the entire `Spec2Prop_ChemistryAfrica` folder into a `.zip` file.
2. Create a new project in Overleaf by uploading the `.zip` file.
3. Overleaf will automatically detect `main_blinded.tex` and `main_nonblind.tex`. Select which one to compile from the Menu > Main document dropdown.
4. Replace the `\fbox` placeholders in the `figures/` directory with your actual diagrams (like `fig1_overall_workflow.pdf`).
5. Verify and update the references tagged with `TBD_VERIFY` in `references.bib`.
