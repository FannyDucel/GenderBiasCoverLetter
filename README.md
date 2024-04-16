# GenderBiasCoverLetter

## Introduction

This repository contains all data and code used in [REF] to run the experiences on gender bias in generated cover letters. 

The goal of these experiments is to generate cover letters with autoregressive language models in French and Italian. In one setting, the prompts do not mention any gender, and in the other setting, the prompt contains one gender marker. We then use a rule-based gender detection system to analyze the gender markers used in the generated text. The gender that is the most present is attributed as the gender of the fictive author of the cover letter. Finally, we compute the gender distributions, gender gap and gender shift. We estimate that a language model is biased if, for a given occupation, the generated texts favor a specific gender over another.

[add schémas]



## Repo organization

### Folders




## How to reproduce the experiments?

`pip install requirements.txt`

### 1. Generate cover letters with LLMs
`python generation_lm.py [model]`, i.e `python generation_lm.py gpt2-fr`.

/!/ Some models require a large amount of VRAM, hence the use of GPUs. We ran them on Grid5k.

Other LLMs can be tested by adding its HuggingFace path #TODO.

### 2. Gender detection
`python gender_detection.py` will create the automatically annotated CSV files.

### 3. Evaluate the gender detection system
`python gender_evaluate.py` returns classification reports based on manual annotations.

### 4. Compute metrics, get figures
Run the various .ipynb files.

