"""Simple exploratory file.
To count the nouns that are in the lexical resources and that are actually used, i.e. detected in the texts
(because over 3000 nouns in the French lexical resources but only 300 in Italian, yet better results in Italian)"""

import pandas as pd
from ast import literal_eval
import json
import glob

def resource_usage(language, detailed_lists=False):
    """Exploratory function that provides the number of gender markers found in the generations in total and the nb
    found in the generations that are in the lexical resources. It shows that only few markers from the lexical resources
    are actually used, so a good detection accuracy could be obtained with smaller resources.

    Parameters
    ----------
    language
        Studied language (fr or it).
    param2
        A bool (False by default), depending on whether or not the user wants more detailed information and the actual sets of gender markers.

    Returns
    -------
    Nothing. Only prints information.
    """
    total_markers = []

    for csv_path in glob.glob(f"../../annotated_texts/{language.upper()}/*/*.csv"):
        df = pd.read_csv(csv_path)

        for i in df["Detailed_markers"]:
            el = i.strip('[]')
            el_list = el.split(", ")
            for el in el_list:
                total_markers.append(el)

    set_total = set(total_markers)

    with open(f"../../data/{language.upper()}/lexical_resources/lexical_res_{language.lower()}.json", encoding="utf-8") as f:
        agents_hum = json.load(f)

    # overlap between the gender markers and the lexical resource (i.e. not count verbs and adjectives)
    from_lexical = [el for el in set_total if el in agents_hum]

    if detailed_lists:
        print("List of elements from the lexical resources that have been found in the generations:", from_lexical)
        print("There are", len(set_total), "gender markers in the generations. Here is the set of them:")
        print(set_total)

    print(f"Among {len(set_total)} total gender markers found in the generations {len(from_lexical)} are in the lexical resources (i.e. without verbs/adjectives found thanks to only Spacy and the rules), out of {len(agents_hum)} entities available in the resource.")

resource_usage("it",False)