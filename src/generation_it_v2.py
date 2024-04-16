import glob
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
torch.cuda.empty_cache()
import json
import sys
#from generation_lm_7b import generate_prompt
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {
   # "gepetto": "LorenzoDeMattei/GePpeTto",
    "xglm-2.9B": "facebook/xglm-2.9B",
    #"gpt2-it":"GroNLP/gpt2-small-italian",
    #"mgpt":"ai-forever/mGPT",
    #"loquace-7b":"cosimoiaia/Loquace-7B",
    #"camoscio-7b":"sag-uniroma2/extremITA-Camoscio-7b",
    "cerbero-7b": "galatolo/cerbero-7b",
    #"fauno-7b": "andreabac3/Fauno-Italian-LLM-7B" #intialisation différente avec peft et llama ?
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

with open("templates_it_genre_fin.json", encoding="utf-8") as f:
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
"""
from vllm import LLM, SamplingParams
sampling_params1 = SamplingParams(top_p=0.75, top_k=100, max_tokens=200)
sampling_params2 = SamplingParams(top_p=0.95, top_k=10, max_tokens=200)
llm = LLM(model=hfpath)  # Name or path of your model
#output = llm.generate("Hello, my name is")
#print(output)

for theme, prompt_list in tqdm(templates.items()):
    outputs1 = llm.generate(prompt_list*3, sampling_params1)
    outputs2 = llm.generate(prompt_list*3, sampling_params2)
    for i, output in enumerate([outputs1, outputs2]):
        with open(f"it_coverletter_{model_name}_vllm.csv", "a", encoding="utf-8") as f:
            if i == 0:
                output_sampling = ["top_p:0.75", "top_k:100", model_name, theme, output.prompt, output.outputs[0].text]
            else:
                output_sampling = ["top_p:0.95", "top_k:10", model_name, theme, output.prompt, output.outputs[0].text]
            writer = csv.writer(f)
            writer.writerows([output_sampling])
"""
prefix = ""
if model_name == "cerbero-7b":
	prefix = "[|Umano|] Scrivere una lettera di presentazione. [|Assistente|] "


for theme, prompt_list in tqdm(templates.items()):
    for i, gen_args_sampling in enumerate([gen_args_sampling1, gen_args_sampling2]) :
        for prompt in prompt_list:
            for _ in range(3):
                answer = generate_prompt(prefix+prompt, model, tokenizer, gen_args_sampling)
                if i==0:
                    output_sampling=["top_p:0.75", "top_k:100", model_name, theme, prompt, answer]
                else:
                    output_sampling=["top_p:0.95", "top_k:10", model_name, theme, prompt, answer]

                with open(f"it_coverletter_{model_name}_genre.csv", "a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows([output_sampling])
