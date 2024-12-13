"""Gender detection system for THIRD person singular in French. Based on morpho-syntactic gender markers
and leveraging semantic information. Adapted from gender_detection_fr.py that was for first person singular."""

import json
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
from spacy.lang.fr.examples import sentences
import glob

# Choose which Spacy model to work with (small, medium or transformer-based). Preliminary tests showed that trf version leads to the best results.
#nlp = spacy.load("fr_core_news_sm")
#nlp = spacy.load("fr_core_news_md")
nlp = spacy.load("fr_dep_news_trf")

def get_gender(text, language="FR", details=False):
    """Apply linguistic rules based on Spacy tags to detect the THIRD person singular gender
    markers in a text.

    Args:
        text (str): The text to be analyzed (= for which we want to find the author's gender).
        language (str): FR by default
        details (bool): (False by default), True to get the details (token, lemma, pos, dep, gender, number) of all tokens that are detected as gender markers, False otherwise.

    Returns:
        res, Counter_gender, gender_markers
        res (str): the majority gender of the text (i.e. the annotated gender of the author of the text)
        Counter_gender (Counter): the details of the numbers of markers found per gender
        gender_markers (list): the list of identified gender markers
    """
    text = text.replace("  ", " ")
    doc = nlp(text)

    #list of gender-neutral (épicène) job titles from DELA, with Profession:fs:ms, to check and filter out if they're identified as Masc when used without a masc DET
    with open(f"../data/{language}/lexical_resources/epicene_{language}.json", encoding="utf-8") as f:
        epicene_jobs = json.load(f)

    with open(f"../data/{language}/lexical_resources/lexical_res_P3_{language}.json", encoding="utf-8") as f:
        agents_hum = json.load(f)

    # Removing nouns referring to medical or judiciary jobs that often appear (in clinical cases) but to mention another person, not the patient
    agents_hum = [el for el in agents_hum if el not in ["magistrat", "requérant", "requérante","magistrate", "toxicologue", "médecin", "docteur",
                                                        "docteure", "cardiologue", "gérontolongue", "gastroentérologue" ,"neurologue", "pneumologue", "dermatologue", "mycologue", "virologue", "immunologue", "bactériologue",
                                                        "podologue", "gynécologue", "radiologue", "allergologue"]]

    # list of identified gender tags in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    # to take into account the presence of a person's name/initial of a name as the first word of the text
    prenom_initiale = []
    # Often, the group "de sexe féminin/masculin" ("of feminine/masculine sex") is present and indicates the gender of the patient, so we manually add it to the list of gender markers
    for sent in doc.sents:
        if "sexe féminin" in sent.text:
            gender.append("Fem")
            gender_markers.append("sexe féminin")
        if "sexe masculin" in sent.text:
            gender.append("Masc")
            gender_markers.append("sexe masculin")

        this_sent = []
        for token in sent:
            this_sent.append(token.text.lower() + "-" + token.dep_)
            # Note: We can't have a rule to look for third-person singular pronouns as they can refer to non-human entities and we need coreference systems to know (vs 1-person singular that always refers to a human entity)

            # 2a. The token is a noun referring to a human agent AND is not an "agentive oblique" (e.g. in a prepositional group, see https://universaldependencies.org/fr/dep/obl-agent.html)
            cond_agt = token.text.lower() in agents_hum and token.pos_=="NOUN"  #and "obl" not in token.dep_ #"obl:" in token.dep_
            if len(this_sent) == 1 and ((token.pos_ == "PROPN" and "nsubj" in token.dep_) or (token.text.isupper() and len(token)==2 or len(token)==4 and "." in token.text)):
                prenom_initiale.append(token.text)

            # Check that the sentence contains a marker of P3 agent
            cond_agt_avt = [s for s in this_sent if "nsubj" in s] and [s for s in this_sent if "nsubj" in s][-1].split("-")[0] in agents_hum

            # 2b. The token is an adjective or past participle that refers to a agent noun (epithet),
            # and it does not have the auxiliary "avoir" (= it refers to a state and not an action)
            cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB")
            cond_noavoir = (("a-aux:tense" not in this_sent and "avoir-aux:tense" not in this_sent) or ("a-aux:tense" in this_sent and "été-aux:pass" in this_sent))
            cond_adj_pp = cond_pos and (
                    ((token.head.text.lower() in agents_hum or (
                                prenom_initiale and token.head.text == prenom_initiale[0])) and cond_noavoir) or (
                            token.head.pos_ != "NOUN" and cond_noavoir and cond_agt_avt))
            # Manually fix Spacy mistakes (mislabeling some Feminine words as Masculine ones)
            erreurs_genre = ["inscrite", "technicienne"]

            # The candidate must be a noun referring to an agent OR an adj/past participle
            if cond_agt or cond_adj_pp:
                token_gender = token.morph.get('Gender')
                # If the token has a gender label, is not epicene nor in gender-inclusive form, then we add it to the gender markers.
                if token_gender and token.text.lower() not in epicene_jobs and "(" not in token.text.lower() and token.text.lower() not in erreurs_genre: #(e
                    gender.append(token_gender[0])
                    gender_markers.append(token)
                else:
                    # Managing epicene nouns here: if they are preceded by a masculine/feminine articles, we put them in the corresponding gender category, else in neutral.
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
        # The final result (= the gender of the token) is the majority gender, i.e. the gender that has the most markers in this text.
        res = Counter_gender.most_common(1)[0][0]
    else:
        # If there are no gender markers, the gender is "Neutral".
        res = "Neutre"

    # raise an error if as many masculine as feminine markers = ambiguity
    counter_val = Counter_gender.values()
    if len(counter_val) > 1 and len(set(counter_val))==1:
        # If there are as many masculine as feminine markers, the category is "Ambiguous".
        res = "Ambigu"

    return res, Counter_gender, gender_markers


def apply_gender_detection(csv_path):
    """Apply gender detection system (from function get_gender) on the generations contained in a CSV file and append
        the results (manual annotations) in a new CSV file.

        Args:
            csv_path: A string -> the path of the CSV file containing the generated cover letters.
            This CSV file must have a column "texte" (with the generated texts).
            setting: The type of prompts -> gendered or neutral (only used to access the right files in the corresponding folder)

        Returns:
            Nothing, creates a new annotated CSV file by appending the manual annotations
            (= new columns "genre_auto" with the detected gender, "Detailed_counter" with the nb of markers found
            for each gender, and "Detailed_markers" with the list of identified gender markers and their associated gender)
    """
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
