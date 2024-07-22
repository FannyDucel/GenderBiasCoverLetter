# GenderBiasCoverLetter

Warning: Repo under construction. Python files are being refactored and enhanced.

The experiment and framework are presented in the following TALN publication: "Évaluation automatique des biais de genre dans des modèles de langue auto-régressif", Fanny Ducel, Aurélie Névéol, Karën Fort. [https://inria.hal.science/hal-04621134/document]?

## Introduction

This repository contains all data and code used in [https://inria.hal.science/hal-04621134/document] to run the experiences on gender bias in generated cover letters. 

The goal of these experiments is to generate cover letters with autoregressive language models in French and Italian. In one setting, the prompts do not mention any gender, and in the other setting, the prompt contains one gender marker. We then use a rule-based gender detection system to analyze the gender markers used in the generated text. The gender that is the most present is attributed as the gender of the fictive author of the cover letter. Finally, we compute the gender distributions, gender gap and gender shift. We estimate that a language model is biased if, for a given occupation, the generated texts favor a specific gender over another.


## Repo organization

- **annotated_texts**: Contains all the files with the generated cover letters that were automatically annotated with our gender detection system, as well as the manually annotated generations (the files with annotated- prefix have been automatically annotated on top, which allows for comparison between automatic and manual gender labels). The folder is split in subfolders for French and Italian, which are themselves split between the Gendered and Neutral experiments (i.e. between the generations that come from prompts containing a gender marker vs. no gender markers).

- **data**: Split into subfolders for French and Italian. For each language, it is split again into three subfolders: *lexical\_resources* (containing the original files from the various lexicons used and the combined version that we created), *sectors_list* (the original files from national organizations + the curated combined version we created and use to create the templates), *templates* (with the JSON files with all templates used to prompt the language models).

- **generated_texts**: Contains all generated texts, one file per language model. Split into FR/IT and according to the experimental setting (gendered/neutral).

- **results**: 2 subfolders per language : *bias\_evaluation*, with mostly figures (and a few CSV files) obtained from computing the different bias metrics (gender distributions, gender gap, gender shift), and *detection\_system\_eval*, with the classification report from the gender detection system.

- **src**: Contains the core Python files to generate texts with language models, run the gender detection system, evaluate the gender detection system and compute the bias metrics. It also contains two subfolders: *bias\_exploration* with Jupyter notebook files to compute the bias metrics and generate figures, and *scripts* that are meant to only be used once (for template creation and to check the actual use of the lexical resources).


## How to reproduce the experiments? (#TODO, in construction)

`pip install requirements.txt`

### 1. Generate cover letters with LLMs
`python src/generation.py [model] [language] [setting]`, i.e `python generation_lm.py gpt2-fr FR neutral`.

/!/ Some models require a large amount of VRAM, hence the use of GPUs. We ran them on Grid5k.

Other LLMs can be tested by adding its HuggingFace path #TODO.

### 2. Use the gender detection system
`python src/gender_detection_fr.py` will create the automatically annotated CSV files for French.

### 3. Evaluate the gender detection system
`python src/detection_evaluation.py` returns classification reports based on manual annotations.

### 4. Compute metrics, get figures
`python src/measure_bias.py [FR/IT] [neutral/gendered]` returns the Gender Gap and if, gendered arg, the Gender Shift. To be used choosing FR or IT, neutral OR gendered, e.g.: `python src/measure_bias.py FR gendered`.
Run the various .ipynb files.
