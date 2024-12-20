"""Run LLM generations"""

import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.cuda.empty_cache()
import json
import sys
from tqdm.auto import tqdm

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
    "croissant":"croissantllm/CroissantLLMBase"
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
with open(f"../data/{language}/templates/{setting}/templates_{setting}_{language}.json", encoding="utf-8") as f:
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

# Iterating to generate for all professional fields, with all combinations of hyperparemeters, for all variations of prompts, and with several iterations for each setting
# Saving the results in a CSV file containing info on the prompt, gender, arguments, ...
for theme, prompt_list in tqdm(templates.items()):
    for i, gen_args_sampling in enumerate([gen_args_sampling1, gen_args_sampling2]) :
        for prompt in prompt_list:
            for _ in range(3):
                answer = generate_prompt(prefix+prompt, model, tokenizer, gen_args_sampling)
                if setting == "gendered":
                    if language == "FR":
                        genre = [
                            'Prompt_fém' if "diplômée" in prompt else 'Prompt_inclusif_parenth' if "diplômé(e)" in prompt else 'Prompt_inclusif_median' if "diplômé·e" in prompt else 'Prompt_masc' if "diplômé" in prompt else '']
                    if language == "IT":
                        genre = [
                            'Prompt_fém' if "laureata" in prompt else 'Prompt_inclusif_parenth' if "laureatə" in prompt else 'Prompt_masc' if "laureato" in prompt else '']
                    if i==0:
                        output_sampling=["top_p:0.75", "top_k:100", model_name, theme, prompt, genre, answer]
                    else:
                        output_sampling=["top_p:0.95", "top_k:10", model_name, theme, prompt, genre, answer]

                else:
                    if i==0:
                        output_sampling=["top_p:0.75", "top_k:100", model_name, theme, prompt, genre, answer]
                    else:
                        output_sampling=["top_p:0.95", "top_k:10", model_name, theme, prompt, genre, answer]

                with open(f"generated_texts/{language}/{setting}_prompts/coverletter_{setting}_{language.lower()}_{model_name}.csv", "a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows([output_sampling])
