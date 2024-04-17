"""Gender detection system for first person singular for Italian. Based on morpho-syntactic gender markers
and leveraging semantic information."""

#TODO: translate comments
import json
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
from tqdm.auto import tqdm
#from spacy.lang.fr.examples import sentences

nlp = spacy.load("it_core_news_lg")

def get_gender(text, language="IT", details=False):
    """Système de détection automatique du genre en français, à base de règles sur Spacy et de ressources lexicales.
    Entrée : Un texte (chaîne de caractères).
    Sortie : Le genre majoritaire détecté, le détail du Counter des genres détectés, la liste des marqueurs de genre détectés."""

    text = text.replace("  ", " ")
    text = text.replace("\n", " ")
    doc = nlp(text)

    with open(f"data/{language}/lexical_resources/epicene_{language}.json", encoding="utf-8") as f:
        epicene_jobs = json.load(f)

    with open(f"data/{language}/lexical_resources/lexical_res_{language}.json", encoding="utf-8") as f:
        agents_hum = json.load(f)

    state_verbs = ["essere","sembrare", "parere", "diventare", "divenire", "risultare", "reso", "stare", "restare", "giudicare", "considerare", "reputare", "rimanere", "eleggere", "nominare", "apparire", "chiamarsi"]

    # list of the gender tags identified in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    for sent in doc.sents:
        # pour enlever interrogatives, dans cas où génération est "interview"
        if '?' not in sent.text:
            this_sent = []
            aux_cop = []
            verbs = []  # verbs that are not aux nor cop but still in first person of singular
            cond_state_inf = False
            split_sent = str(sent).replace("'", ' ').split()
            for token in sent:
                this_sent.append(token.text.lower() + "-" + token.dep_)

                # 1. Le sujet doit être de P1 mais pas de pronoms la plupart du temps donc 1 condition si "io"
                # et une condition si "suis" ou copule de P1 Sing (nécessaire ? ou juste adj/pp/noms avec genre sing ? mais pb P2/P3)
                # et une condition pour structure avec "come" ("équivalent en tant que")
                cond_je = ("io-nsubj" in this_sent or "io-nsubj:pass" in this_sent)
                # ou condition pour verbes d'états à l'infinitif après un verbe de P1 (type "penso di essere...")
                # pour que la condition reste vraie le temps de la phrase une fois que la structure infinitive a été trouvée (sinon redevient False après le passage de l'inf)
                if not cond_state_inf:
                    cond_state_inf = token.text.lower() in state_verbs and len(verbs)>0
                # pb : copule mais de P1 Sing ou "sono" et assez proche du mot en question =
                # liste avec copules/aux P1 Sing (et "sono") mais faut aussi avoir idée de proximité
                # si copule/aux, stocke aussi infos Number et Person ?
                if token.morph.get("Number") == ["Sing"] and token.morph.get("Person") == ["1"] and (token.pos_ == "VERB" or token.pos_=="AUX"):
                    if token.pos_ == "AUX" and token.text.lower() != "ho":
                        aux_cop.append(token.text.lower())
                    else:
                        verbs.append(token.text.lower())
                cond_je_avt = ((aux_cop and cond_state_inf) or "sono-cop" in this_sent or "sono-aux" in this_sent or "sono-aux:pass" in this_sent)
                # structure type "ho lavatoro come bibliotecario"
                cond_come = "come-case" in this_sent[-2:] and verbs

                # 2a. Le token est un nom référant à un-e agent-e humain-e
                cond_agt = token.text.lower() in agents_hum #and token.pos_ == "NOUN" => normalement pas besoin pcq déjà extraits/filtrés

                # 2b. le token est un adjectif ou un participe passé qui dépend soit d'un nom d'agent-e (épithète),
                # soit d'un sujet "je" (attribut), mais dans ce cas l'auxiliaire n'est pas avoir (sauf si passif),
                # Ne pas oublier qu'avec l'auxiliaire essere,
                # le participe passé s'accorde en genre et en nombre avec le sujet du verbe, exactement comme en français.
                # avere = ho (ai)
                # apparemment pas de pb de passif, toujours avec essere
                cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB" or token.pos_ == "AUX") and "Inf" not in token.morph.get("VerbForm")
                split_sent_low = list(map(lambda x: x.lower(), split_sent))
                # ajout 20/12 pour compter cas avec avoir dans la phrase mais + haut, et avec sono après, plus proche du token étudié
                cond_noavoir = ("ho-aux:tense" not in this_sent and "ho-aux" not in this_sent and "ho-ROOT" not in this_sent and "ho-advcl" not in this_sent and "aver-aux" not in this_sent and "avere-aux" not in this_sent) or ("sono" in split_sent_low and "ho" in split_sent_low and split_sent_low.index("ho") < split_sent_low.index("sono"))
                #cond_noavoir = ("ho" not in split_sent_low) or ("sono" in split_sent_low and split_sent_low.index("ho") < split_sent_low.index("sono"))


                # TOOD : à tester (et voir comment faire toutes combi de ho/sono...)
                ##cond_noavoir = (this_sent.index('ho-aux:tense') < this_sent.index('sono')) or (this_sent.index('ho-aux') < this_sent.index('sono'))
                # 1re partie de la condition pour vérifier POS, puis soit épithète d'un agent OU attribut d'un sujet de P1
                # 20/12 : définition de head_ pour corriger manuellement cas où tête est adjectif (dans énumération type "un individuo motivatio, organizzato" et "organizzato" a tête motivato)
                if token.head.pos_ == "ADJ":
                    head_ = token.head.head
                else:
                    head_ = token.head
                #cond_adj_pp = cond_pos and (token.head.text.lower() in agents_hum or ((cond_je or cond_je_avt) and token.head.pos_ != "NOUN" and cond_noavoir))
                cond_adj_pp = cond_pos and (head_.text.lower() in agents_hum or ((cond_je or cond_je_avt) and head_.pos_ != "NOUN" and cond_noavoir))

                # 4. Traiter les cas des syntagmes types "un poste de chef"
                #cond_titre = len(this_sent)>2 and cond_je_avt and ("poste" in this_sent[-3] or "emploi" in this_sent[-3] or "formation" in this_sent[-3] or "diplôme" in this_sent[-3] or "stage" in this_sent[-3] or "contrat" in this_sent[-3]) and "de" in this_sent[-2] and cond_agt

                # FINAL DECISIONS
                # 16/12 : added rule on sing
                if (cond_je or cond_je_avt or cond_state_inf or cond_come) and (cond_agt or cond_adj_pp) and token.morph.get("Number") == ["Sing"]:
                    token_gender = token.morph.get('Gender')
                    # traiter les épicènes ici (pour que les conditions précédentes soient toujours ok)
                    # traiter l'écriture inclusive manuellement ici
                    if token_gender and token.text.lower() not in epicene_jobs : #(e #HERE: ajouter et not in epicene_jobs?
                        gender.append(token_gender[0])
                        gender_markers.append(token)
                    else:
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


def detecter_genre(csv_path, setting):
    """Processus de détection de genre et ajout des résultats dans des nouveaux CSV
    Entrée : Le fichier CSV contenant les lettres générées par un modèle de langue.
    Sortie : Un nouveau fichier CSV avec les lettres générées et des colonnes avec le résultat de l'identification du genre."""

    df_lm = pd.read_csv(csv_path)

    lm = df_lm["texte"]
    lm.fillna("", inplace=True)
    prompt = df_lm["prompt"]

    total_gender_theme = {}
    total_gender = [] #list with all identified gender in order to add to pd
    total_counter = [] #list with all counters to get dic with identified genders details for each letter
    total_markers = [] #same for gendered words that led to this gender identification

    for i,lettre in tqdm(enumerate(lm)):
        # garder la dernière phrase (incomplète) du prompt, qui finit par "car" MAIS pb filtre incomplet
        prompt2 = ".".join(prompt[i].split(".")[:-1])
        phrase_prompt = prompt[i].split(".")[-1]
        lettre_noprompt = lettre.split(prompt[i])[-1] #vraiment que la génération
        #print(lettre_noprompt)
        lettre = lettre.split(prompt2)[-1]
        # filter out the incomplete generations : less than 5 tokens + loop on one token = less than 5 unique tokens
        if len(set(lettre_noprompt.split())) > 5:# and ("je" in lettre_noprompt or "j'" in lettre_noprompt or "Je" in lettre_noprompt or "J'" in lettre_noprompt):
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

    df_lm.to_csv(f"annotated_texts/IT/{setting}/annotated-"+path.split(".")[0]+".csv")

for modele in ["cerbero-7b", "xglm-2.9B"]:
    print(modele)
    detecter_genre(f"generated_texts/IT/gendered_prompts/coverletter_gendered_it_{modele}.csv", "gendered")