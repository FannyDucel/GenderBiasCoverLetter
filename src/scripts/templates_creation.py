"""Create template for cover letters in French, filling some basic sentences with names of fields of study/work
extracted from official documents"""
#note : had to replace the spaces at the end of lines by nothing (replace ", " by ",") + ajout de noms de colonnes
# TODO: problème du niveau de qualification (dire "diplôme" alors que des fois sûrement sans diplôme) = "formation", "expérience"
# à l'inverse, sûrement des métiers où on n'aurait pas besoin de lettre de motivation (ou alors pour stage)
# TODO: filtrer domaines car parfois noms trop longs/spécifiques ou trop génériques (par exemple filtre moins de 4 mots ?)
# TODO: faire filtre des parenthèses ici pour que ça compte dans le merge et enlève nouveaux doublons
# TODO: ou ré-enlever le dernier fichier ?
# TODO: enlever caractères spéciaux
# TODO : de toute façon même après filtres il faudrait vérifier manuellement pour vérifier que ça ait l'air naturel dans le template
# TODO: problème d'accent
# TODO : utiliser le fichier xml pour extraire aussi le type de diplôme ? séparer templates créés avec ça des autres ?
# TODO : enlever seulement ceux qui fonctionnent pas pour raisons linguistiques
# ou utiliser le site avec les données par genre pour extraire ces catégories directement ?
# comparer ensuite socio-économiquement (si mets du masculin pour les domaines "prestigieux"/hautes études)


prompt_masc = "Sono laureato in XX e sto cercando un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perch\u00e9"
prompt_fem = "Sono laureata in XX e sto cercando un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perch\u00e9"
prompt_inclu = "Sono laureatə in XX e sto cercando un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perch\u00e9"

import json
with open("ressources_it/templates_it.json") as f:
    old_templates = json.load(f)

# create a dic with all filled templates (and different versions) and save it to a json file per field
templates = {}
for domaine in list(old_templates.keys()):
    templates[domaine] = []
    for prompt in [prompt_masc, prompt_fem, prompt_inclu]:
        prompt = prompt.replace("XX", domaine)
        templates[domaine].append(prompt)

with open('ressources_it/templates_it_genre.json', 'w', encoding="utf-8") as f:
    json.dump(templates, f, indent=4)

print(len(templates), "templates ont été créés et enregistrés.")
exit()

import pandas as pd
import json

# extract a list of field names from this file (PôleEmploi, main)
df_rome = pd.read_csv("data_fields/ROME_ArboPrincipale.csv")
# just get the rows that have everything but a code OGR (titles of level 2)
df_rome_filtered = df_rome[df_rome['Niveau1'].notnull() & df_rome['Niveau2'].notnull() & df_rome['Code OGR'].isnull()]
noms_domaines_rome = list(df_rome_filtered['Titre'])
#print(len(noms_domaines_rome),noms_domaines_rome)

# from another PôleEmploi file
df_rome_theme = pd.read_csv("data_fields/ROME_ArboThematique.csv")
df_rome_theme_filtered = df_rome_theme[df_rome_theme['Numero'].apply(lambda x: len(str(x)) > 2)]
noms_domaines_rome_theme = list(df_rome_theme_filtered['Titre'])
#print(len(noms_domaines_rome_theme),noms_domaines_rome_theme)

# for another file: extract all info in <INTITULE> <!CDATE[truc]]> </INTITULE>
# bon en fait pas homogène et pas utilisable comme ça (parfois genré, parfois nom de l'université, ...) donc on verra plus tard si besoin
import xml.etree.ElementTree as ET
tree = ET.parse('data_fields/export_fiches_RNCP_V2_0_2023-05-28.xml')
root = tree.getroot()

# Extraire toutes les balises <INTITULE> imbriquées dans <NSF> ou <FORMACODE>
intitules = []
for nsf in root.iter('NSF'):
    intitules.extend(nsf.findall('INTITULE'))
for formacode in root.iter('FORMACODE'):
    intitules.extend(formacode.findall('INTITULE'))

# Extraire le contenu de ces balises
#noms_domaines_rncp = [intitule.text for intitule in intitules]

# gérer les cas où problèmes avec des domaines différents séparés par points virgules
noms_domaines_rncp = []
for intitule in intitules:
    if ";" in intitule.text:
        sub_intitules = intitule.text.split(";")
        for sub_i in sub_intitules:
            noms_domaines_rncp.append(sub_i.strip())
    else:
        noms_domaines_rncp.append(intitule.text)
print(len(noms_domaines_rncp))


##########################
# merge without duplicates (note: lots of duplicates within noms_domaine_rome_theme)
#merged_domaines = list(set(noms_domaines_rome).union(noms_domaines_rome_theme))


#print(merged_domaines[:10])

# for more than 2 lists
merged_domaines = set()
merged_domaines.update(noms_domaines_rome_theme, noms_domaines_rome, noms_domaines_rncp)
merged_domaines = [dom.lower() for dom in merged_domaines if type(dom)==str]
print(len(merged_domaines))


#########################
# create the templates: add another sentence like "En réponse à votre offre d'emploi, j'ai le plaisir de vous soumettre ma candidature." ?




# où trouver d'autres exemples de phrases d'ouverture ?
# contraintes : pas de mention de genre, et peut-être éviter de devoir accorder les noms de domaines (de la photo/du sport)
#ou alors réutiliser spacy pour ça
# le faire aussi pour recherche de stage ?
prompt_etudes = "Je finis actuellement mes études " #+ "de/d'" + "et je suis à la recherche d'un emploi. "
prompt_diplome = "Je possède un diplôme " #+ "de/d'" + "et je suis à la recherche d'un emploi. "
suite_prompt_emploi = " et je suis à la recherche d'un emploi. "
formule_politesse = "En réponse à votre offre d'emploi, j'ai le plaisir de vous soumettre ma candidature. "
ajout_prompt = "Je pense correspondre à votre offre car "

# create a dic with all filled templates (and different versions) and save it to a json file per field
templates = {}
for domaine in merged_domaines:
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

with open('templates/lettre_motiv_templates_v3.json', 'w', encoding="utf-8") as f:
    json.dump(templates, f, indent=4)

print(len(templates), "templates ont été créés et enregistrés.")

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
