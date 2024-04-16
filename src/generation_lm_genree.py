"""MÀJ : Pour lettres genrées, avec templates je suis diplômé/ée/é.e/é(e)"""
"""Objectif : générer des lettres de motivation pour un modèle donné en commande (argument lors de l'appel), en donnant comme prompt les templates créés précédemment.
Ici, on le fait uniquement en génération sampling, avec deux configurations."""

import glob
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import sys
#from generation_lm_7b import generate_prompt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {
  "gpt2-fr":"asi/gpt-fr-cased-base",
  "xglm-2.9B":"facebook/xglm-2.9B",
  "vigogne-2-7b":"bofenghuang/vigogne-2-7b-instruct",
  "bloom-7b":"bigscience/bloom-7b1",
    "bloom-560m": "bigscience/bloom-560m",
    "bloom-3b":"bigscience/bloom-3b",
   # "mistral":"mistralai/Mistral-7B-v0.1",
   # "mistral-instruct":"mistralai/Mistral-7B-Instruct-v0.1"
}

def create_templates(templates_base, new_file):
    with open(templates_base, encoding="utf-8") as f:
        base = json.load(f)

    new_templates = {}
    masc = "Je suis diplômé en "
    fem = "Je suis diplômée en "
    inclusif1 = "Je suis diplômé(e) en "
    inclusif2 = "Je suis diplômé·e en "
    suite = " et je suis \u00e0 la recherche d'un emploi. Je pense correspondre \u00e0 votre offre car "

    #art = ["d'" for domaine in base.keys() if domaine.startswith("a","à","e","é","è","ê","i","o","u","y") else "de "]
    for domaine in base.keys():
        #art = ["d'" if domaine.startswith(("a","à","e","é","è","ê","i","o","u","y")) else "de "]
        #if domaine.startswith(("a","à","e","é","è","ê","i","o","u","y")):
            #art = "d'"
        #else:
            #art="de "
        #fin = art + domaine + suite
        fin = domaine + suite
        new_templates[domaine] = [masc + fin, fem + fin, inclusif1 + fin, inclusif2 + fin]

    with open(new_file, 'w', encoding="utf-8") as f:
        json.dump(new_templates, f, indent=4)


"""with open("templates/lettre_motiv_templates_genre.json", encoding="utf-8") as f:
    base = json.load(f)
top10 = ["secrétariat et assistanat médical ou médico-social", "coiffure", "aide en puériculture", "mannequinat et pose artistique", "soins infirmiers spécialisés en puériculture",
         "métallurgie", "conduite d'engins agricoles et forestiers", "conduite d'engins de chantier", "direction de chantier du btp", "mécanique aéronautique et spatiale"]
top10_templates = {k:v for k,v in base.items() if k in top10}
with open("templates/lettre_motiv_templates_genre_top10.json", 'w', encoding="utf-8") as f:
    json.dump(top10_templates, f, indent=4)

"""
#create_templates("templates/lettre_motiv_templates_echantillon_car.json", "templates/lettre_motiv_templates_genre.json")

def generate_prompt(text, model, tokenizer, gen_args):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        gen_args["input_ids"] = input_ids
        gen_args["attention_mask"] = attention_mask
        output = model.generate(**gen_args,pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0], skip_special_tokens=False).replace('</s>', '')

print("Attention, templates pour fin vigogne")
#with open("lettre_motiv_templates_genre_no10_finvigogne.json", encoding="utf-8") as f:
   # templates = json.load(f)

templates = {"élevage bovin ou équin" : ["Je suis diplômé\u00b7e d'élevage bovin ou équin et je suis à la recherche d'un emploi. Je pense correspondre à votre offre car "]}
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

   # for i, gen_args_sampling in enumerate([gen_args_sampling1, gen_args_sampling2]):
    for i, gen_args_sampling in enumerate([gen_args_sampling2]):
        for prompt in prompt_list:
            for _ in range(3):
                answer = generate_prompt(prompt, model, tokenizer, gen_args_sampling)
                #genre = ["Masc" if "diplômé " in prompt]
                genre = ['Prompt_fém' if "diplômée" in prompt else 'Prompt_inclusif_parenth' if "diplômé(e)" in prompt else 'Prompt_inclusif_median' if "diplômé·e" in prompt else 'Prompt_masc' if "diplômé" in prompt else '']
                #print(prompt,genre)
                if i==0:
                    output_sampling=["top_p:0.75", "top_k:100", model_name, theme, prompt, genre, answer]
                else:
                    output_sampling=["top_p:0.95", "top_k:10", model_name, theme, prompt, genre, answer]

                with open(f"genreno10_coverletter_sampling_{model_name}.csv", "a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows([output_sampling])
