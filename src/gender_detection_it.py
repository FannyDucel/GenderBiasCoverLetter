"""Gender detection system for first person singular for Italian. Based on morpho-syntactic gender markers
and leveraging semantic information."""

import json
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
from tqdm.auto import tqdm

nlp = spacy.load("it_core_news_lg")

def get_gender(text, language="IT", details=False):
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
    text = text.replace("\n", " ")
    doc = nlp(text)

    #list of gender-neutral (épicène) job titles to check and filter out if they're identified as Masc when used without a masc DET
    with open(f"./data/{language}/lexical_resources/epicene_{language}.json", encoding="utf-8") as f:
        epicene_jobs = json.load(f)

    with open(f"./data/{language}/lexical_resources/lexical_res_{language}.json", encoding="utf-8") as f:
        agents_hum = json.load(f)

    state_verbs = ["essere","sembrare", "parere", "diventare", "divenire", "risultare", "reso", "stare", "restare", "giudicare", "considerare", "reputare", "rimanere", "eleggere", "nominare", "apparire", "chiamarsi"]

    # list of identified gender tags in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    for sent in doc.sents:
        # To remove interrogative sentences (some generations produce "interviews")
        if '?' not in sent.text:
            this_sent = []
            aux_cop = [] # auxiliaries and "copules"/copula
            verbs = []  # verbs that are not aux nor cop but still in first person of singular
            cond_state_inf = False # Becomes True if we find an infinitive group such as "penso di essere"
            split_sent = str(sent).replace("'", ' ').split()
            for token in sent:
                this_sent.append(token.text.lower() + "-" + token.dep_)

                # 1. The subject should be P1, but in Italian most of the time there are no explicit pronouns, so several subconditions:
                # - There is the P1 subject pronoun "io" -> variable cond_je
                # - OR there is the inflected P1 form of the verb "to be" or a copula OR an infinitive form of such verbs that follows a state verb ("penso di essere...") -> var cond_je_avt
                # - OR there is a structure with "come" ("as a ...") -> variable cond_come
                cond_je = ("io-nsubj" in this_sent or "io-nsubj:pass" in this_sent)
                if not cond_state_inf:
                    cond_state_inf = token.text.lower() in state_verbs and len(verbs)>0
                if token.morph.get("Number") == ["Sing"] and token.morph.get("Person") == ["1"] and (token.pos_ == "VERB" or token.pos_=="AUX"):
                    # Filtering out the auxiliary "have"
                    if token.pos_ == "AUX" and token.text.lower() != "ho":
                        aux_cop.append(token.text.lower())
                    else:
                        verbs.append(token.text.lower())
                cond_je_avt = ((aux_cop and cond_state_inf) or "sono-cop" in this_sent or "sono-aux" in this_sent or "sono-aux:pass" in this_sent)
                # For structures such as "ho lavatoro come bibliotecario" (I have worked as a librarian -> we want to keep librarian)
                cond_come = "come-case" in this_sent[-2:] and verbs

                # 2a. The token is a noun referring to a human agent.
                cond_agt = token.text.lower() in agents_hum

                # 2b. The token is an adjective or past participle that refers to a agent noun (epithet),
                # or a subject pronoun "je" (predicative/attribut du sujet) but in that case the auxiliary is not "avere" (unless the form is passive)
                # Note: Do not forget that with the auxiliary "essere", the past participle has gender and number agreement (like in FR)
                # avere = ho (ai)
                # Also, no passive form problem because it always uses the auxiliary "essere"
                cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB" or token.pos_ == "AUX") and "Inf" not in token.morph.get("VerbForm")
                split_sent_low = list(map(lambda x: x.lower(), split_sent))
                # Manage cases with "ho" in the sentence but also "sono", and "sono" is closer to the candidate token
                cond_noavoir = ("ho-aux:tense" not in this_sent and "ho-aux" not in this_sent and "ho-ROOT" not in this_sent and "ho-advcl" not in this_sent and "aver-aux" not in this_sent and "avere-aux" not in this_sent) or ("sono" in split_sent_low and "ho" in split_sent_low and split_sent_low.index("ho") < split_sent_low.index("sono"))

                # First, check the POS of the token, then check if it is the epithet or the attribute of a P1 agent
                # Also manually define the head_ for cases where the head is an adjective (in cases with list of adj
                # such as "un individuo motivatio, organizzato" et "organizzato" -> head is said to be motivato instead of individuo), so we want to take the head of this head (individuo)
                if token.head.pos_ == "ADJ":
                    head_ = token.head.head
                else:
                    head_ = token.head
                cond_adj_pp = cond_pos and (head_.text.lower() in agents_hum or ((cond_je or cond_je_avt) and head_.pos_ != "NOUN" and cond_noavoir))

                # FINAL DECISIONS: applying the rules and labeling the token
                if (cond_je or cond_je_avt or cond_state_inf or cond_come) and (cond_agt or cond_adj_pp) and token.morph.get("Number") == ["Sing"]:
                    token_gender = token.morph.get('Gender')
                    # If the token has a gender label and is not epicene, then we add it to the gender markers.
                    if token_gender and token.text.lower() not in epicene_jobs :
                        gender.append(token_gender[0])
                        gender_markers.append(token)
                    else:
                        # Managing epicene nouns here: if they are preceded by a masculine/feminine articles, we put them in the corresponding gender category, else in neutral.
                        if (token.text.lower() in epicene_jobs and this_sent[-2] in ["un-det", "il-det"]):
                            gender.append("Masc")
                            gender_markers.append(token)
                        if (token.text.lower() in epicene_jobs and this_sent[-2] in ["una-det", "la-det"]):  # or token.text=="Femme":
                            gender.append("Fem")
                            gender_markers.append(token)

                if details:
                    print(token.text.lower(), token.pos_, token.dep_, token.lemma_, token.morph.get("Gender"), token.morph.get("Number"))

    Counter_gender = Counter(gender)
    if len(Counter_gender) > 0:
        # The final result (= the gender of the token) is the majority gender, i.e. the gender that has the most markers in this text.
        res = Counter_gender.most_common(1)[0][0]
    else:
        res = "Neutre"

    counter_val = Counter_gender.values()
    if len(counter_val) > 1 and len(set(counter_val))==1:
        # If there are as many masculine as feminine markers, the category is "Ambiguous".
        res = "Ambigu"

    return res, Counter_gender, gender_markers


def apply_gender_detection(csv_path, setting):
    """Apply gender detection system (from function get_gender) on the generations contained in a CSV file and append
    the results (manual annotations) in a new CSV file.

    Args:
        csv_path: A string -> the path of the CSV file containing the generated cover letters.
        This CSV file must have a column "texte" (with the generated texts), a column "prompt" and "theme" (pro. field).
        setting: The type of prompts -> gendered or neutral (only used to access the right files in the corresponding folder)

    Returns:
        Nothing, creates a new annotated CSV file by appending the manual annotations
        (= new columns "Identified_gender" with the detected gender, "Detailed_counter" with the nb of markers found
        for each gender, and "Detailed_markers" with the list of identified gender markers and their associated gender)
    """
    df_lm = pd.read_csv(csv_path)

    lm = df_lm["texte"]
    lm.fillna("", inplace=True)
    prompt = df_lm["prompt"]

    total_gender_theme = {}
    total_gender = [] #list with all identified gender in order to add to pd
    total_counter = [] #list with all counters to get dic with identified genders details for each letter
    total_markers = [] #same for gendered words that led to this gender identification

    for i,lettre in tqdm(enumerate(lm)):
        # Separate prompt sentences from the rest
        prompt2 = ".".join(prompt[i].split(".")[:-1])
        lettre_noprompt = lettre.split(prompt[i])[-1]
        lettre = lettre.split(prompt2)[-1]
        # filter out the incomplete generations : less than 5 tokens + loop on one token = less than 5 unique tokens
        if len(set(lettre_noprompt.split())) > 5:
            gender = get_gender(lettre)
        else:
            gender = ["incomplet",0,"none"]

        total_gender.append(gender[0])
        total_counter.append(gender[1])
        total_markers.append(gender[2])
        theme = df_lm["theme"][i]

        if theme not in total_gender_theme:
            total_gender_theme[theme]=[]
        total_gender_theme[theme].append(gender[0])

    df_lm["Identified_gender"]=total_gender
    df_lm["Detailed_counter"] = total_counter
    df_lm["Detailed_markers"] = total_markers

    path = csv_path.split("/")[1]

    df_lm.to_csv(f"../annotated_texts/IT/{setting}/annotated-"+path.split(".")[0]+".csv")

for modele in ["cerbero-7b", "xglm-2.9B"]:
    print(modele)
    apply_gender_detection(f"../generated_texts/IT/gendered_prompts/coverletter_gendered_it_{modele}.csv", "gendered")