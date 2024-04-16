"""Objectif : générer des lettres de motivation pour un modèle donné en commande (argument lors de l'appel), en donnant comme prompt les templates créés précédemment.
Ici, on le fait uniquement en génération sampling, avec deux configurations."""

import glob
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {
  "gpt2-fr":"asi/gpt-fr-cased-base",
  "xglm-2.9B":"facebook/xglm-2.9B",
  "vigogne-2-7b":"bofenghuang/vigogne-2-7b-instruct",
  "bloom-7b":"bigscience/bloom-7b1",
 "falcon-7b":"tiiuae/falcon-7b"
}


def generate_prompt(text, model, tokenizer, gen_args):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        gen_args["input_ids"] = input_ids
        gen_args["attention_mask"] = attention_mask
        output = model.generate(**gen_args)
        return tokenizer.decode(output[0], skip_special_tokens=False).replace('</s>', '')

with open("lettre_motiv_templates_echantillon_car_p2.json", encoding="utf-8") as f:
    templates = json.load(f)

output = [] # List of lists

# Load tokenizer and model outside the loops
tokenizer_dict = {}
model_dict = {}

#model_name from command
model_name = sys.argv[1]
hfpath = models[model_name]

print(model_name)

tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(hfpath, cache_dir="./my_model_directory/")
model_dict[model_name] = AutoModelForCausalLM.from_pretrained(hfpath, cache_dir="./my_model_directory/",trust_remote_code=True)
model_dict[model_name].to(device)

output_sampling = []

for theme, prompt_list in templates.items():
    tokenizer = tokenizer_dict[model_name]
    model = model_dict[model_name]
    gen_args_sampling1 = {
            "do_sample": True,
            "top_p": 0.75,
            "top_k": 100,
            # "temperature":temp/10,
            "max_new_tokens": 200,
        }
    gen_args_sampling2 = {
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 10,
            # "temperature":temp/10,
            "max_new_tokens": 200,
        }

    for i, gen_args_sampling in enumerate([gen_args_sampling1, gen_args_sampling2]):
        for prompt in prompt_list:
            for _ in range(3):
                answer = generate_prompt(prompt, model, tokenizer, gen_args_sampling)
                if i==0:
                    output_sampling=["top_p:0.75", "top_k:100", model_name, theme, prompt, answer]
                else:
                    output_sampling=["top_p:0.95", "top_k:10", model_name, theme, prompt, answer]

                with open(f"coverletter_sampling_{model_name}.csv", "a") as f:
                    writer = csv.writer(f)
                    writer.writerows([output_sampling])
