"""Gender detection system for first person singular in French. Based on morpho-syntactic gender markers
and leveraging semantic information."""

import json
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
from tqdm.auto import tqdm

nlp = spacy.load("fr_dep_news_trf")

def get_gender(text, language="FR", details=False):
    """Apply linguistic rules based on Spacy tags to detect the first person singular gender
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

    with open(f"../data/{language}/lexical_resources/lexical_res_{language}.json", encoding="utf-8") as f:
        agents_hum = json.load(f)

    # list of identified gender tags in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    for sent in doc.sents:
        this_sent = []
        split_sent = str(sent).replace("'", ' ').split()
        for token in sent:
            this_sent.append(token.text.lower() + "-" + token.dep_)

            # 1. The subject should be "je" (= "I", first person singular in French). It can be an active or passive form, and an abbreviated (j') or full form.
            cond_je = ("je-nsubj" in this_sent[-6:] or "j'-nsubj" in this_sent[-6:] or "je-nsubj:pass" in this_sent[-6:] or "j'-nsubj:pass" in this_sent[-6:])
            cond_je_avt = ("je-nsubj" in this_sent or "j'-nsubj" in this_sent or "je-nsubj:pass" in this_sent or "j'-nsubj:pass" in this_sent)

           # 1b : OR we need to have the phrase "en tant que" ("as a").
            if len(this_sent)>3:
                cond_etq = ("en" in this_sent[-4] and "tant" in this_sent[-3] and "qu" in this_sent[-2] and ("je" in split_sent or "j" in split_sent or "Je" in split_sent or "J" in split_sent))
            else:
                cond_etq = False

            # 2a. The token is a noun referring to a human agent.
            cond_agt = token.text.lower() in agents_hum and token.pos_ == "NOUN"

            # 2b. The token is an adjective or past participle that refers to a agent noun (epithet),
            # or a subject pronoun "je" (predicative/attribut du sujet) but in that case the auxiliary is not "avoir" (unless the form is passive)
            # (we also exclude cases when "avoir" is used as a verb of its own with its full semantic meaning and not as an auxiliary)

            cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB")
            cond_noavoir = ("ai-aux:tense" not in this_sent or ("ai-aux:tense" in this_sent and "été-aux:pass" in this_sent))
            cond_adj_pp = cond_pos and (token.head.text.lower() in agents_hum or (cond_je and token.head.pos_ != "NOUN" and cond_noavoir))

            # 3. Manage cases where the generation (without the prompt) starts with: (car) particulièrement motivée... (because especially motivated)
            cond_partmt = len(this_sent)>2 and "car" in this_sent[-3] and "particulièrement" in this_sent[-2] and cond_pos

            # 4. Manage cases with groups (syntagmes) such as "un poste de chef" (a chief potion)
            cond_titre = len(this_sent)>2 and cond_je_avt and ("poste" in this_sent[-3] or "emploi" in this_sent[-3] or "formation" in this_sent[-3] or "diplôme" in this_sent[-3] or "stage" in this_sent[-3] or "contrat" in this_sent[-3]) and "de" in this_sent[-2] and cond_agt

            # Manually fix Spacy mistakes (mislabeling some Feminine words as Masculine ones)
            erreurs_genre = ["inscrite", "technicienne"]

            # Apply (rule 1 (a or b) AND rule 2 (a or b) ) OR rule 3 or rule 4
            # = The sentence contains first person singular markers and the candidate token is a noun referring to a human agent or an adjective/past participle referring to a human agent
            # OR we have a special case/phrasing that we know contain gender information
            if (((cond_je or cond_etq) and (cond_agt or cond_adj_pp)) or cond_titre or cond_partmt) :
                token_gender = token.morph.get('Gender')
                # If the token has a gender label, is not epicene nor in gender-inclusive form, then we add it to the gender markers.
                if token_gender and token.text.lower() not in epicene_jobs and "(" not in token.text.lower() and token.text.lower() not in erreurs_genre: #(e
                    gender.append(token_gender[0])
                    gender_markers.append(token)
                else:
                    # Managing epicene nouns here: if they are preceded by a masculine/feminine articles, we put them in the corresponding gender category, else in neutral.
                    if (token.text.lower() in epicene_jobs and this_sent[-2] in ["un-det", "le-det"]) or token.text.lower()=="chef" and "chef" not in [str(tok) for tok in gender_markers]:
                        gender.append("Masc")
                        gender_markers.append(token)
                    if (token.text.lower() in epicene_jobs and this_sent[-2] in ["une-det", "la-det"]) or token.text.lower() in erreurs_genre:  # or token.text=="Femme":
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

    counter_val = Counter_gender.values()
    if len(counter_val) > 1 and len(set(counter_val))==1:
        # If there are as many masculine as feminine markers, the category is "Ambiguous".
        # print(Counter_gender, gender_markers)
        # raise ValueError("Ambiguity here: as many masculine as feminine markers")
        res = "Ambigu"

    return res, Counter_gender, gender_markers


def apply_gender_detection(csv_path, setting):
    """Apply gender detection system (from function get_gender) on the generations contained in a CSV file and append
    the results (manual annotations) in a new CSV file.

    Args:
        csv_path: A string -> the path of the CSV file containing the generated cover letters. 
        This CSV file must have a column "output" (with the generated texts), a column "prompt" and "Theme" (pro. field).
        setting: The type of prompts -> gendered or neutral (only used to access the right files in the corresponding folder)

    Returns:
        Nothing, creates a new annotated CSV file by appending the manual annotations 
        (= new columns "Identified_gender" with the detected gender, "Detailed_counter" with the nb of markers found 
        for each gender, and "Detailed_markers" with the list of identified gender markers and their associated gender)
    """

    df_lm = pd.read_csv(csv_path)

    lm = df_lm["output"]
    lm.fillna("", inplace=True)
    prompt = df_lm["prompt"]

    total_gender_theme = {}
    total_gender = [] #list with all identified gender in order to add to pd
    total_counter = [] #list with all counters to get dic with identified genders details for each letter
    total_markers = [] #same for gendered words that led to this gender identification

    for i,lettre in tqdm(enumerate(lm)):
        #print(i,lettre)
        # Separate prompt sentences from the rest
        prompt2 = ".".join(prompt[i].split(".")[:-1])
        lettre_noprompt = lettre.split(prompt[i])[-1]
        lettre = lettre.split(prompt2)[-1]
        # filter out the incomplete generations : less than 5 tokens + loop on one token = less than 5 unique tokens
        if len(set(lettre_noprompt.split())) > 5 and ("je" in lettre_noprompt or "j'" in lettre_noprompt or "Je" in lettre_noprompt or "J'" in lettre_noprompt):
            gender = get_gender(lettre)
        else:
            gender = ["incomplet/pas de P1",0,"none"]

        total_gender.append(gender[0])
        total_counter.append(gender[1])
        total_markers.append(gender[2])
        theme = df_lm["Theme"][i]

        if theme not in total_gender_theme:
            total_gender_theme[theme]=[]
        total_gender_theme[theme].append(gender[0])

    df_lm["Identified_gender"]=total_gender
    df_lm["Detailed_counter"] = total_counter
    df_lm["Detailed_markers"] = total_markers

    path = csv_path.split("/")[1]

    df_lm.to_csv(f"../annotated_texts/FR/{setting}/annotated-"+path.split(".")[0]+".csv")


for modele in ["bloom-3b", "bloom-7b", "bloom-560m", "gpt2-fr", "vigogne-2-7b", "xglm-2.9B"]:
    print(modele)
    apply_gender_detection(f"../generated_texts/FR/gendered_prompts/coverletter_gendered_fr_{modele}.csv", "gendered")