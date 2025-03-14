<h1>Clinical rule analysis </h1>
<p>
  <img src="https://img.shields.io/badge/license-mit-blue.svg">
  <img src="https://img.shields.io/badge/python-3.9--3.11-blue">
  <img src="https://img.shields.io/badge/numpy-<2.0-darkgreen">
</p>

Code for analyzing clinical decision rules and supporting the analysis underlying a systematic review on bias in CDIs (<a href="https://www.medrxiv.org/content/10.1101/2025.02.12.25320965v1">Obra et al. 2025</a>) <br/>
<img align="center" width=100% src="https://csinva.io/clinical-rule-analysis/results/mdcalc_fig2.svg?sanitize=True&kill_cache=1"> </img>


### Abstract

> Clinical decision instruments (CDIs) face an equity dilemma. On the one hand, they often reduce disparities in patient care through data-driven standardization of best practices. On the other hand, this standardization may itself inadvertently perpetuate bias and inequality within healthcare systems. Here, we quantify different measures of potential for implicit bias present in CDI development that can inform future CDI development. We find evidence for systematic bias in the development of 690 CDIs that underwent validation through various analyses: self-reported participant demographics are skewed—e.g. 73% of participants are White, 55% are male; investigator teams are geographically skewed—e.g. 52% in North America, 31% in Europe; CDIs use predictor variables that may be prone to bias—e.g. 13 CDIs explicitly use Race and Ethnicity; outcome definitions may further introduce bias—e.g. 28% of CDIs involve follow-up, which may disproportionately skew outcome representation based on socioeconomic status. As CDIs become increasingly prominent in medicine, we recommend that these factors are considered during development and clearly conveyed to clinicians using CDIs.



# Files
- [notebooks_data_prep](notebooks_data_prep): Contains notebooks used to clean and prepare the data
- [notebooks_llm](notebooks_llm): Contains notebooks used for LLM experiments
- [data](data): Contains the raw data, processed data, and data dictionary
  - (Optional) Paper pdfs are put into this [gdrive folder](https://drive.google.com/drive/folders/1OUXtsddxEAOl3tKEZegBQSwArUecb-6J).
    - Each paper is named using its `id`
    - If using rclone, download with `rclone ls gdrive:data/public_data/mdcalc-papers --drive-shared-with-me`
    - To download a single paper, `rclone copy gdrive:data/public_data/mdcalc-papers/10129.pdf . --drive-shared-with-me`


# Working with the cleaned data
- Raw data is contained as a dataframe in the [data/data_clean.pkl](data/data_clean.pkl) file.
  - A semi-complete data dictionary is given in [data/data_clean_dictionary.json](data/data_clean_dictionary.json).
