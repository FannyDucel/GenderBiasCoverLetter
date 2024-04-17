"""Adaptation for third person singular for French."""

import json
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
from spacy.lang.fr.examples import sentences
import glob

#nlp = spacy.load("fr_core_news_sm")
#nlp = spacy.load("fr_core_news_md")
nlp = spacy.load("fr_dep_news_trf")

def get_gender(text, language="FR", details=False):
    """Système de détection automatique du genre en français, à base de règles sur Spacy et de ressources lexicales.
    Entrée : Un texte (chaîne de caractères).
    Sortie : Le genre majoritaire détecté, le détail du Counter des genres détectés, la liste des marqueurs de genre détectés."""

    # remplacer les doubles espaces, car ils peuvent fausser la détection spacy
    text = text.replace("  ", " ")
    doc = nlp(text)

    #list of gender-neutral (épicène) job titles from DELA, with Profession:fs:ms, to check and filter out if they're identified as Masc when used without a masc DET
    # enlever erreurs : notamment suffixes en -eur "procureur", "professeur", "proviseur", censeur, chauffeur, chef, auteur, docteur, défenseur, gouverneur, ingénieur, ...
    # ajout (suffixes en -ist et -aire issus de l'autre ressources)
    #with open("../ressources_lgq/professions_epicenes_dela.json", encoding="utf-8") as f:
    with open(f"data/{language}/lexical_resources/epicene_{language}.json", encoding="utf-8") as f:
        epicene_jobs = json.load(f)

    with open(f"data/{language}/lexical_resources/lexical_res_P3_{language}.json", encoding="utf-8") as f:
        agents_hum = json.load(f)

    # enlever les professions médicales ou juridiques qui apparaissent souvent mais pour désigner une personne tierce, pas lae patient-e
    agents_hum = [el for el in agents_hum if el not in ["magistrat", "requérant", "requérante","magistrate", "toxicologue", "médecin", "docteur",
                                                        "docteure", "cardiologue", "gérontolongue", "gastroentérologue" ,"neurologue", "pneumologue", "dermatologue", "mycologue", "virologue", "immunologue", "bactériologue",
                                                        "podologue", "gynécologue", "radiologue", "allergologue"]]

    # list of the gender tags identified in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    # pour présence de prénom/initiale(s) comme premier mot du texte => à réessayer avec condition que ce soit .head()
    prenom_initiale = []
    for sent in doc.sents:
        if "sexe féminin" in sent.text:
            gender.append("Fem")
            gender_markers.append("sexe féminin")
        if "sexe masculin" in sent.text:
            gender.append("Masc")
            gender_markers.append("sexe masculin")

        this_sent = []
        split_sent = str(sent).replace("'", ' ').split()
        for token in sent:
            this_sent.append(token.text.lower() + "-" + token.dep_)
            # NON-1. Le sujet doit être "il" ou "elle" = morph:Gender=Fem|Number=Sing|Person=3, pos:PRON, dep:nsubj
            # En fait non, pb de coréférence (il peut référer à un examen ou n'importe quoi d'autre de mentionné) + pbs si prénoms comme seul élément de réf au genre
            #cond_iel = ("il-nsubj" in this_sent or "elle-nsubj" in this_sent or "il-nsubj:pass" in this_sent or "elle'-nsubj:pass" in this_sent)
            #cond_p3 = (token.pos_ == "PRON" and token.morph.get("Person") == "['3']" and token.morph.get("Number") == "['Sing']")

            # 2a. Le token est un nom référant à un-e agent-e humain-e et n'est pas un argument "oblique" (dans un syntagme prépositionnel) #obl:agent, obl:arg
            cond_agt = token.text.lower() in agents_hum and token.pos_=="NOUN"  #and "obl" not in token.dep_ #"obl:" in token.dep_
            if len(this_sent) == 1 and ((token.pos_ == "PROPN" and "nsubj" in token.dep_) or (token.text.isupper() and len(token)==2 or len(token)==4 and "." in token.text)):
                prenom_initiale.append(token.text)

            # Vérifier qu'il y ait un marqueur de P3 agent dans la phrase (seulement agent de la ressource)
            #cond_p3_agt_avt = ("il-nsubj" in this_sent or "elle-nsubj" in this_sent or "il-nsubj:pass" in this_sent or "elle'-nsubj:pass" in this_sent) or ([s for s in this_sent if "nsubj" in s] and [s for s in this_sent if "nsubj" in s][-1] in agents_hum)
            cond_agt_avt = [s for s in this_sent if "nsubj" in s] and [s for s in this_sent if "nsubj" in s][-1].split("-")[0] in agents_hum

            # 2b. le token est un adjectif ou un participe passé qui dépend soit d'un nom d'agent-e (épithète),
            # soit d'un sujet "je" (attribut), mais dans ce cas l'auxiliaire n'est pas avoir (sauf si passif),
            # ((on exclut également les cas où "avoir" est utilisé en tant que verbe et non d'auxiliaire ("sémantiquement plein")))
            cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB")
            cond_noavoir = (("a-aux:tense" not in this_sent and "avoir-aux:tense" not in this_sent) or ("a-aux:tense" in this_sent and "été-aux:pass" in this_sent))
            #cond_adj_pp = cond_pos and ((token.head.text.lower() in agents_hum and cond_noavoir and token.head.dep_ != "obl:arg") or (token.head.pos_ != "NOUN" and cond_noavoir and cond_agt_avt))
            #cond_adj_pp = cond_pos and (
                        #(((token.head.text.lower() in agents_hum and "avait-aux:tense" not in this_sent[-3:]) or (prenom_initiale and token.head.text == prenom_initiale[0])) and cond_noavoir) or (
                            #token.head.pos_ != "NOUN" and cond_noavoir and cond_agt_avt))
            cond_adj_pp = cond_pos and (
                    ((token.head.text.lower() in agents_hum or (
                                prenom_initiale and token.head.text == prenom_initiale[0])) and cond_noavoir) or (
                            token.head.pos_ != "NOUN" and cond_noavoir and cond_agt_avt))
            # corriger manuellement des problèmes de Spacy qui met comme Masc des mots Fem
            erreurs_genre = ["inscrite", "technicienne"]

            if cond_agt or cond_adj_pp:
                token_gender = token.morph.get('Gender')
                # traiter les épicènes ici (pour que les conditions précédentes soient toujours ok)
                # traiter l'écriture inclusive manuellement ici
                if token_gender and token.text.lower() not in epicene_jobs and "(" not in token.text.lower() and token.text.lower() not in erreurs_genre: #(e
                    gender.append(token_gender[0])
                    gender_markers.append(token)
                else:
                    if (token.text.lower() in epicene_jobs and len(this_sent)>1 and this_sent[-2] in ["un-det", "le-det"]) or token.text.lower()=="chef" and "chef" not in [str(tok) for tok in gender_markers]:
                        gender.append("Masc")
                        gender_markers.append(token)
                    if (token.text.lower() in epicene_jobs and len(this_sent)>1 and this_sent[-2] in ["une-det", "la-det"]) or token.text.lower() in erreurs_genre:  # or token.text=="Femme":
                        gender.append("Fem")
                        gender_markers.append(token)
                    if "(" in token.text.lower():
                        gender.append("Neutre")
                        gender_markers.append(token)

            if details:
                print(token.text.lower(), token.pos_, token.dep_, token.lemma_, token.morph.get("Gender"), token.morph.get("Number"))

    Counter_gender = Counter(gender)
    if len(Counter_gender) > 0:
        res = Counter_gender.most_common(1)[0][0]
    else:
        res = "Neutre"

    # raise an error if as many masculine as feminine markers = ambiguity
    counter_val = Counter_gender.values()
    if len(counter_val) > 1 and len(set(counter_val))==1:
        # print(Counter_gender, gender_markers)
        # raise ValueError("Ambiguity here: as many masculine as feminine markers")
        res = "Ambigu"

    return res, Counter_gender, gender_markers


def detecter_genre(csv_path):
    """Processus de détection de genre et ajout des résultats dans des nouveaux CSV
    Entrée : Le fichier CSV contenant les lettres générées par un modèle de langue.
    Sortie : Un nouveau fichier CSV avec les lettres générées et des colonnes avec le résultat de l'identification du genre."""

    df_lm = pd.read_csv(csv_path)

    lm = df_lm["texte"]
    lm.fillna("", inplace=True)

    total_gender = [] #list with all identified gender in order to add to pd
    total_counter = [] #list with all counters to get dic with identified genders details for each letter
    total_markers = [] #same for gendered words that led to this gender identification

    for lettre in tqdm(lm):
        gender = get_gender(lettre)
        total_gender.append(gender[0])
        total_counter.append(gender[1])
        total_markers.append(gender[2])

    df_lm["genre_auto"]=total_gender
    df_lm["Detailed_counter"] = total_counter
    df_lm["Detailed_markers"] = total_markers


    df_lm.to_csv(csv_path.split(".")[0]+f"_P3.csv")
