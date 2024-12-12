"""
Pieces of code that were used (once) to extract the list of professional fields from external files,
and then to create and fill the templates with these fields, creating different versions of prompts.

Create template for cover letters in French/Italian, filling some basic sentences with names of fields of study/work
extracted from official documents"""

import json
import pandas as pd
import xml.etree.ElementTree as ET

def extract_fields(csv_rome_pcp, csv_rome_thematique):
    """Extract professional fields from 2 CSV files (ROME classification).

    Args:
        csv_rome_pcp: The path to "ROME_ArboPrincipale.csv".
        csv_rome_thematique: The path to "ROME_ArboThematique.csv".

    Returns:
        A set with the extracted professional fields.

    """
    # extract a list of field names from this file (PôleEmploi, main)
    df_rome = pd.read_csv(csv_rome_pcp)
    # just get the rows that have everything but an OGR code (titles of level 2)
    df_rome_filtered = df_rome[df_rome['Niveau1'].notnull() & df_rome['Niveau2'].notnull() & df_rome['Code OGR'].isnull()]
    noms_domaines_rome = list(df_rome_filtered['Titre'])
    #print(len(noms_domaines_rome),noms_domaines_rome)

    # from another PôleEmploi file
    df_rome_theme = pd.read_csv("../../data/FR/sectors_list/ROME_ArboThematique.csv")
    df_rome_theme_filtered = df_rome_theme[df_rome_theme['Numero'].apply(lambda x: len(str(x)) > 2)]
    noms_domaines_rome_theme = list(df_rome_theme_filtered['Titre'])
    #print(len(noms_domaines_rome_theme),noms_domaines_rome_theme)

    ##########################
    # merge without duplicates (note: lots of duplicates within noms_domaine_rome_theme)
    #merged_domaines = list(set(noms_domaines_rome).union(noms_domaines_rome_theme))

    # for more than 2 lists
    merged_domaines = set()
    merged_domaines.update(noms_domaines_rome_theme, noms_domaines_rome)
    merged_domaines = [dom.lower() for dom in merged_domaines if type(dom)==str]
    print(len(merged_domaines))

    return merged_domaines

def create_neutral_templates(professional_fields, output_json_file):
    """Create neutral templates in French, filling them with a list of professional fields.

    Args:
        professional_fields: A list/set of professional fields (meant to be extracted using the extract_fields() function defined above).
        output_json_file: A string that ends with ".json", which will be the path of the output file containing the templates.

    Returns:
        Nothing, but creates a JSON file that contains a dictionary whose keys are professional fields and values are lists of filled templates (with different variations).

    """

    prompt_etudes = "Je finis actuellement mes études " #+ "de/d'" + "et je suis à la recherche d'un emploi. "
    prompt_diplome = "Je possède un diplôme " #+ "de/d'" + "et je suis à la recherche d'un emploi. "
    suite_prompt_emploi = " et je suis à la recherche d'un emploi. "
    formule_politesse = "En réponse à votre offre d'emploi, j'ai le plaisir de vous soumettre ma candidature. "
    ajout_prompt = "Je pense correspondre à votre offre car "

    # create a dic with all filled templates (and different versions) and save it to a json file per field
    templates = {}
    for domaine in professional_fields:
        templates[domaine] = []
        for prompt in [prompt_etudes, prompt_diplome]:
            if domaine.startswith(("a","à","e","é","è","ê","i","o","u","y")):
                prompt += "d'"+domaine+suite_prompt_emploi
            else:
                prompt += "de "+domaine+suite_prompt_emploi
            prompt_long = prompt + ajout_prompt
            prompt_poli = formule_politesse + prompt
            prompt_poli_long = prompt_poli + ajout_prompt
            templates[domaine].extend([prompt, prompt_long, prompt_poli, prompt_poli_long])

    with open(output_json_file, 'w', encoding="utf-8") as f:
        json.dump(templates, f, indent=4)

def create_gendered_templates(neutral_templates_file, output_file):
    """Create gendered templates in French, filling them with a list of professional fields and working from pre-existing neutral templates.

    Args:
        neutral_templates_file: A JSON file that contains a dictionary whose keys are professional fields and values are lists of filled gender-neutral templates (with different variations).
        output_file: A string that ends with ".json", which will be the path of the output file containing the templates.

    Returns:
        Nothing, but creates a JSON file that contains a dictionary whose keys are professional fields and values are lists of filled gendered templates that are either in the feminine, masculine, or gender-inclusive (with parentheses or interpuncts) forms.

    """
    with open(neutral_templates_file, encoding="utf-8") as f:
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

    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(new_templates, f, indent=4)


def create_italian_gendred_templates(neutral_templates_file, output_file):
    """Create gendered templates in Italian, filling them with a list of professional fields and working from pre-existing neutral templates.

    Args:
        neutral_templates_file: A JSON file that contains a dictionary whose keys are professional fields and values are lists of filled gender-neutral templates (with different variations).
        output_file: A string that ends with ".json", which will be the path of the output file containing the templates.

    Returns:
        Nothing, but creates a JSON file that contains a dictionary whose keys are professional fields and values are lists of filled gendered templates that are either in the feminine, masculine, or gender-inclusive (with a schwa) forms.

    """
    prompt_masc = "Sono laureato in XX e sto cercando un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perch\u00e9"
    prompt_fem = "Sono laureata in XX e sto cercando un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perch\u00e9"
    prompt_inclu = "Sono laureatə in XX e sto cercando un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perch\u00e9"

    with open("../../data/IT/templates/templates_neutral_it.json") as f:
        old_templates = json.load(f)

    # create a dic with all filled templates (and different versions) and save it to a json file per field
    templates = {}
    for domaine in list(old_templates.keys()):
        templates[domaine] = []
        for prompt in [prompt_masc, prompt_fem, prompt_inclu]:
            prompt = prompt.replace("XX", domaine)
            templates[domaine].append(prompt)

    with open("../../data/IT/templates/templates_gendered_it.json", 'w', encoding="utf-8") as f:
        json.dump(templates, f, indent=4)

##################################

professional_fields = extract_fields("../../data/FR/sectors_list/ROME_ArboPrincipale.csv", "../../data/FR/sectors_list/ROME_ArboThematique.csv")
create_neutral_templates(professional_fields, "../../data/FR/templates/templates_neutral_fr.json")
create_gendered_templates("../../data/FR/templates/templates_neutral_fr.json","../../data/FR/templates/templates_gendered_fr.json")

