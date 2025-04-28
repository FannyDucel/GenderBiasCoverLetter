"""Run LLM generations"""

import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.cuda.empty_cache()
import json
import sys
from tqdm.auto import tqdm
import os
from huggingface_hub import login
from vllm import LLM
from vllm.sampling_params import SamplingParams

login("your_token")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# put models path (on HuggingFace)
models = {
    "xglm-2.9B": "facebook/xglm-2.9B",
    "cerbero-7b": "galatolo/cerbero-7b",
    "gpt2-fr":"asi/gpt-fr-cased-base",
    "vigogne-2-7b":"bofenghuang/vigogne-2-7b-instruct",
    "bloom-7b":"bigscience/bloom-7b1",
    "bloom-560m": "bigscience/bloom-560m",
    "bloom-3b":"bigscience/bloom-3b",
    "croissantbase":"croissantllm/CroissantLLMBase",
    "croissant-it":"croissantllm/CroissantLLMChat-v0.1",
    "gemma-2-2b":"google/gemma-2-2b",
    "gemma-2-2b-it":"google/gemma-2-2b-it",
    "llama-3.2-3b":"meta-llama/Llama-3.2-3B",
    "llama-3.2-3b-it":"meta-llama/Llama-3.2-3B-Instruct"
    }

def generate_prompt(text, model, tokenizer, gen_args):
    """" Triggers the generation of a text replying to a prompt.

    Args:
        text (str): The text to be continued (= the prompt).
        model: The LM to use for the generation.
        tokenizer: The tokenizer to use for the generation and tokenization of the prompt (usually the tokenizer associated to the chosen LM).
        gen_args (dict): The chosen generation arguments (top_p, top_k, nb of tokens, ...)

    Returns:
        The generated text (after decoding).
    """
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt')
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        gen_args["input_ids"] = input_ids
        gen_args["attention_mask"] = attention_mask
        output = model.generate(**gen_args)
        return tokenizer.decode(output[0], skip_special_tokens=False).replace('</s>', '')

# Use the args given by the user when calling the script
language = sys.argv[2]
setting = sys.argv[3]
## adapt the path to abord executions issues
relative_filepath = f"./data/{language}/templates/templates_{setting}_{language.lower()}.json"
absolute_filepath = os.path.abspath(relative_filepath)
with open(absolute_filepath, encoding="utf-8") as f:
    templates = json.load(f)

output = [] # List of lists

# Load tokenizer and model outside the loops
tokenizer_dict = {}
model_dict = {}

#model_name from command
model_name = sys.argv[1]
hfpath = models[model_name]

print(model_name)
if model_name == "mistral-base-3.1-24b" or model_name == "mistral-instruct-3.1-24b-it":
    model_dict[model_name] = LLM(
        model=hfpath,
        tokenizer_mode="mistral", 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(hfpath, cache_dir="./my_model_directory/")
    model_dict[model_name].to(device)

else:
    tokenizer_dict[model_name] = AutoTokenizer.from_pretrained(hfpath, cache_dir="./my_model_directory/")
    model_dict[model_name] = AutoModelForCausalLM.from_pretrained(hfpath, cache_dir="./my_model_directory/", trust_remote_code=True)
    model_dict[model_name].to(device)

output_sampling = []

tokenizer = tokenizer_dict[model_name]
model = model_dict[model_name]
gen_args_sampling1 = {
    "do_sample": True,
    "top_p": 0.75,
    "top_k": 100,
    "max_new_tokens": 200,
}
gen_args_sampling2 = {
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 10,
    "max_new_tokens": 200,
}

prefix = ""
# As cerbero is an instruct model, we need a "pre-prompt":
if model_name == "cerbero-7b":
	prefix = "[|Umano|] Scrivere una lettera di presentazione. [|Assistente|] "

if model_name == "mistral-base-3.1-24b" or model_name == "mistral-instruct-3.1-24b-it":
    sampling_params = SamplingParams(
        temperature=0.15,
        max_tokens=200,
    )


output_file = f"generated_texts/{language}/{setting}_prompts/coverletter_{setting}_{language.lower()}_{model_name}.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)


def determine_genre(prompt, language):
    """Determine the genre of the prompt based on the language and the prompt content.
    Args:
        prompt (str): The prompt text.
        language (str): The language of the prompt ("FR" or "IT").
    Returns:
        list: A list containing the genre of the prompt.
    """
    if language == "FR":
        return [
            'Prompt_fém' if "diplômée" in prompt else
            'Prompt_inclusif_parenth' if "diplômé(e)" in prompt else
            'Prompt_inclusif_median' if "diplômé·e" in prompt else
            'Prompt_masc' if "diplômé" in prompt else ''
        ]
    elif language == "IT":
        return [
            'Prompt_fém' if "laureata" in prompt else
            'Prompt_inclusif_parenth' if "laureatə" in prompt else
            'Prompt_masc' if "laureato" in prompt else ''
        ]

#Setting the first line of CSV file
if setting == "gendered":
    columns = ["", "top_p", "top_k", "modele", "theme", "prompt", "genre_prompt", "output"]
elif setting == "neutral":
    columns = ["", "top_p", "top_k", "modele", "theme", "prompt", "output"]
else:
    print("Error Setting not recognized")

with open(output_file, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)

# Iterating to generate for all professional fields, with all combinations of hyperparemeters, for all variations of prompts, and with several iterations for each setting
# Saving the results in a CSV file containing info on the prompt, gender, arguments, ...
index = 0
for theme, prompt_list in tqdm(templates.items()):
    for i, gen_args_sampling in enumerate([gen_args_sampling1, gen_args_sampling2]):
        for prompt in prompt_list:
            #Generation of 3 identical prompts 
            for _ in range(3):
                answer = generate_prompt(prefix + prompt, model, tokenizer, gen_args_sampling)
                if setting == "gendered":
                    genre = determine_genre(prompt, language)
                    if i==0:
                        output_sampling=[index, "top_p:0.75", "top_k:100", model_name, theme, prompt, genre, answer]
                    else:
                        output_sampling=[index, "top_p:0.95", "top_k:10", model_name, theme, prompt, genre, answer]
                elif setting == "neutral":
                    if i==0:
                        output_sampling=[index, "top_p:0.75", "top_k:100", model_name, theme, prompt, answer]
                    else:
                        output_sampling=[index, "top_p:0.95", "top_k:10", model_name, theme, prompt, answer]
                else: 
                    print("Error Setting not recognized")

                with open(output_file, "a", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(output_sampling)

                index += 1