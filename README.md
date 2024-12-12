# GenderBiasCoverLetter

Warning: Repo under construction. Python files are being refactored and enhanced.

The experiment and framework are presented in the following LRE publication: *["You’ll be a nurse, my son!" Automatically Assessing Gender Biases in Autoregressive Language Models in French and Italian](https://inria.hal.science/hal-04803403)*. Fanny Ducel, Aurélie Névéol, Karën Fort. Language Resources and Evaluation. 

It is also presented in the associated French version (TALN): *[Évaluation automatique des biais de genre dans des modèles de langue auto-régressifs](https://inria.hal.science/hal-04621134/document)*, Fanny Ducel, Aurélie Névéol, Karën Fort. 

If you have questions or issues, feel free to contact me at *fanny.ducel@universite-paris-saclay.fr*

## Introduction

This repository contains all data and code used in <https://inria.hal.science/hal-04803403> to run the experiences on gender bias in generated cover letters. 

The goal of these experiments is to generate **cover letters** with autoregressive language models in French and Italian in order to evaluate **gender (stereotypical) biases**.

In one setting, the prompts (i.e. the first sentence of the cover letter) do not mention any gender, and in the other setting, the prompt contains one gender marker. W

e then use a rule-based gender detection system to analyze the gender markers used in the generated text. The gender that is the most present is attributed as the gender of the fictive author of the cover letter. 

Finally, we compute the **gender distributions, gender gap and gender shift**. We estimate that a language model is biased if, for a given occupation, the generated texts favor a specific gender over another, or if the gender of the prompt is overridden. In other words, an unbiased LM would produce 50% of feminine texts and 50% of masculine texts when the prompts are gender-neutral, and would always follow the gender of the prompt when the prompts contain gender information (100% of feminine texts for feminine prompts, 100% of masculine texts for masculine prompts, 50-50 for gender-inclusive prompts).


## Repo organization

- **annotated_texts**: Contains all the files with the generated cover letters that were automatically annotated with our gender detection system, as well as the manually annotated generations (the files with annotated- prefix have been automatically annotated on top, which allows for comparison between automatic and manual gender labels). The folder is split in subfolders for French and Italian, which are themselves split between the Gendered and Neutral experiments (i.e. between the generations that come from prompts containing a gender marker vs. no gender markers).

- **data**: Split into subfolders for French and Italian. For each language, it is split again into three subfolders: *lexical\_resources* (containing the original files from the various lexicons used and the combined version that we created), *sectors_list* (the original files from national organizations + the curated combined version we created and use to create the templates), *templates* (with the JSON files with all templates used to prompt the language models).

- **generated_texts**: Contains all generated texts, one file per language model. Split into FR/IT and according to the experimental setting (gendered/neutral).

- **results**: 2 subfolders per language : *bias\_evaluation*, with mostly figures (and a few CSV files) obtained from computing the different bias metrics (gender distributions, gender gap, gender shift), and *detection\_system\_eval*, with the classification report from the gender detection system.

- **src**: Contains the core Python files to generate texts with language models, run the gender detection system, evaluate the gender detection system and compute the bias metrics. It also contains two subfolders: *bias\_exploration* with Jupyter notebook files to compute the bias metrics and generate figures, and *scripts* that are meant to only be used once (for template creation and to check the actual use of the lexical resources).

Most Python functions of .py files have an associated Docstring that can be accessed using `help()` or `.__doc__`. 

## How to reproduce the experiments?

`pip install requirements.txt`

### 1. Generate cover letters with LLMs
`python src/generation.py [model] [language] [setting]`, i.e `python generation.py gpt2-fr FR neutral`.

/!/ Some models require a large amount of VRAM, hence the use of GPUs. We ran them on Grid5k.

Other LLMs can be tested by adding their HuggingFace path #TODO.

### 2. Use the gender detection system
`python src/gender_detection_fr.py` will create the automatically annotated CSV files for French (i.e. each cover letter is annotated with the gender of its putative author).

### 3. Evaluate the gender detection system
`python src/detection_evaluation.py` returns classification reports based on manual annotations.

### 4. Compute bias metrics, get figures
`python src/measure_bias.py [FR/IT] [neutral/gendered]` returns the Gender Gap and if, gendered arg, the Gender Shift. To be used choosing FR or IT, neutral OR gendered, e.g.: `python src/measure_bias.py FR gendered`.
You can also run the various .ipynb files for more detailed information and to generate figures.
