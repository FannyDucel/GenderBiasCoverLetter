"""To count the nouns that are in the lexical resources and that are actually used, i.e. detected in the texts
(because over 3000 nouns in the French lexical resources but only 300 in Italian, yet better results in Italian)"""

import pandas as pd
from ast import literal_eval
import json
import glob

total_markers = []

for csv_path in glob.glob("gender_output/coverletter_detection_v2_*"):
    df = pd.read_csv(csv_path)

    for i in df["Detailed_markers"]:
        el = i.strip('[]')
        el_list = el.split(", ")
        for el in el_list:
            total_markers.append(el)

set_total = set(total_markers)
print(len(set_total))
print(set_total)

with open("ressources_lgq/inclu_fem_demonette_semcor_union_nc_hum-agt-pers_corr.json", encoding="utf-8") as f:
    agents_hum = json.load(f)

# overlap between the gender markers and the lexical resource (i.e. not count verbs and adjectives)
from_lexical = [el for el in set_total if el in agents_hum]
print(from_lexical)
print(len(from_lexical))


total_markers = []

for csv_path in glob.glob("ressources_it/gender_output_it/it_coverletter*"):
    df = pd.read_csv(csv_path)

    for i in df["Detailed_markers"]:
        el = i.strip('[]')
        el_list = el.split(", ")
        for el in el_list:
            total_markers.append(el)

set_total = set(total_markers)
print(len(set_total))
print(set_total)

with open("ressources_it/livingner_mustshe_mats_corr.json") as f:
    agents_hum = json.load(f)

# overlap between the gender markers and the lexical resource (i.e. not count verbs and adjectives)
from_lexical = [el for el in set_total if el in agents_hum]
print(from_lexical)
print(len(from_lexical))