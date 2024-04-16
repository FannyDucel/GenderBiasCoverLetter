"""But : détection automatique du genre utilisé dans les lettres de motivation générées
-> Compter le nombre de masc/fém par profession
Ajouter le détail dans le csv : genre identifié (majoritaire), dico détail du genre id, dico détails des indices"""
import json

import pandas as pd

"""Test to try and do gender identification with spacy (morphologizer)
so that when the cover letters will be generated in French we can automatically detect the gender it used
(accord en genre des adjectifs, participes passés, etc)
=> a bit simplistic for now"""
import spacy
from collections import Counter
from tqdm import tqdm
from spacy.lang.fr.examples import sentences

#nlp = spacy.load("fr_core_news_sm")
#nlp = spacy.load("fr_core_news_md")
nlp = spacy.load("fr_dep_news_trf")

def get_gender_v7b(text, details=False):
    """Reprendre du début, version plus minimaliste fondée sur ressources lexicales :
    avec juste condi_je, listes noms et adj/pp dont head est un token dans liste de nom ou je"""
    # remplacer les doubles espaces, car ils peuvent fausser la détection spacy
    text = text.replace("  ", " ")
    #text = text.replace(" ès ", " ")
    #text = text.replace(" ème ", " ")
    doc = nlp(text)

    #list of gender-neutral (épicène) job titles from DELA, with Profession:fs:ms, to check and filter out if they're identified as Masc when used without a masc DET
    # enlever erreurs : notamment suffixes en -eur "procureur", "professeur", "proviseur", censeur, chauffeur, chef, auteur, docteur, défenseur, gouverneur, ingénieur, ...
    # ajout (suffixes en -ist et -aire issus de l'autre ressources)
    with open("ressources_lgq/epicenes_corr.json", encoding="utf-8") as f:
        epicene_jobs = json.load(f)
        epicene_jobs.append("tout")
        epicene_jobs.append("toute")
    #epicene_jobs = ["tout", "toute", "personne"]

    # /!/ "personne", "petit", "secteur", "milieu", "aide", "soutien", "professionnel(le)", "ordinateur", "(arrière)-(petit)-enfant", "titulaire"
    # enlevés manuellement des listes combinées
    # ajouté : salarié, salariée, salarié(e,
    ##with open("ressources_lgq/demonette_semcor_union_nc_hum-agt-pers.json", encoding="utf-8") as f:
    #with open("ressources_it/fr/livingner_mustshe_mats_fr.json", encoding="utf-8") as f:
    with open("ressources_lgq/inclu_fem_demonette_semcor_union_nc_hum-agt-pers_corr.json", encoding="utf-8") as f:
        agents_hum = json.load(f)
    #agents_hum = ["patate","tomate"]

    #print("magistrat" in agents_hum)

    # list of the gender tags identified in the adj/verbs of the text
    gender = []
    # list of the tokens that have a gender tag (and are adj/verbs)
    gender_markers = []
    for sent in doc.sents:
        this_sent = []
        split_sent = str(sent).replace("'", ' ').split()
        for token in sent:
            this_sent.append(token.text.lower() + "-" + token.dep_)

            # 1. Le sujet doit être "je" (à mettre + propre peut-être)
            cond_je = ("je-nsubj" in this_sent[-6:] or "j'-nsubj" in this_sent[-6:] or "je-nsubj:pass" in this_sent[-6:] or "j'-nsubj:pass" in this_sent[-6:])
            cond_je_avt = ("je-nsubj" in this_sent or "j'-nsubj" in this_sent or "je-nsubj:pass" in this_sent or "j'-nsubj:pass" in this_sent)

           # 1b : ou, on doit retrouver la formule "en tant que"
            # ou cas fréquent de la formule "en tant que/qu'"
            if len(this_sent)>3:
                    #test pour ajouter contrainte d'un pronom de P1 dans la phrase mais pb du "j'" mal découpé par un simple split
                cond_etq = ("en" in this_sent[-4] and "tant" in this_sent[-3] and "qu" in this_sent[-2] and ("je" in split_sent or "j" in split_sent or "Je" in split_sent or "J" in split_sent))

            else:
                cond_etq = False

            # 2a. Le token est un nom référant à un-e agent-e humain-e
            cond_agt = token.text.lower() in agents_hum and token.pos_ == "NOUN" #=> normalement pas besoin pcq déjà extraits/filtrés

            # 2b. le token est un adjectif ou un participe passé qui dépend soit d'un nom d'agent-e (épithète),
            # soit d'un sujet "je" (attribut), mais dans ce cas l'auxiliaire n'est pas avoir (sauf si passif),
            # ((on exclut également les cas où "avoir" est utilisé en tant que verbe et non d'auxiliaire ("sémantiquement plein")))

            cond_pos = (token.pos_ == "ADJ" or token.pos_ == "VERB")
            cond_noavoir = ("ai-aux:tense" not in this_sent or ("ai-aux:tense" in this_sent and "été-aux:pass" in this_sent))
            cond_adj_pp = cond_pos and (token.head.text.lower() in agents_hum or (cond_je and token.head.pos_ != "NOUN" and cond_noavoir))
            #cond_adj_pp = cond_pos and (token.head.text.lower() in agents_hum or ((cond_je or cond_me) and token.head.pos_ != "NOUN" and cond_noavoir))

            # 3. Traiter les cas où la génération (en retirant le prompt) commence par : (car) particulièrement motivée...
            cond_partmt = len(this_sent)>2 and "car" in this_sent[-3] and "particulièrement" in this_sent[-2] and cond_pos

            # 4. Traiter les cas des syntagmes types "un poste de chef"
            cond_titre = len(this_sent)>2 and cond_je_avt and ("poste" in this_sent[-3] or "emploi" in this_sent[-3] or "formation" in this_sent[-3] or "diplôme" in this_sent[-3] or "stage" in this_sent[-3] or "contrat" in this_sent[-3]) and "de" in this_sent[-2] and cond_agt

            # 5. ABANDONNÉ Phrases qui ne contiennent pas "je" mais des pronoms de P1 objet ("Mes compétences me permettent d'être...)
            #cond_me = "me-iobj" in this_sent[-6:] or "m'-iobj" in this_sent[-6:]

            # corriger manuellement des problèmes de Spacy qui met comme Masc des mots Fem
            erreurs_genre = ["inscrite", "technicienne"]

            #if (((cond_je or cond_etq or cond_me) and (cond_agt or cond_adj_pp)) or cond_titre or cond_partmt) :
            if (((cond_je or cond_etq) and (cond_agt or cond_adj_pp)) or cond_titre or cond_partmt) :
                token_gender = token.morph.get('Gender')
                # traiter les épicènes ici (pour que les conditions précédentes soient toujours ok)
                # traiter l'écriture inclusive manuellement ici
                if token_gender and token.text.lower() not in epicene_jobs and "(" not in token.text.lower() and token.text.lower() not in erreurs_genre: #(e
                    gender.append(token_gender[0])
                    gender_markers.append(token)
                #else:
                    #gender.append(gender[-1])
                else:
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


def add_gender_to_csv_update(version,csv_path,modele,golden_gender_header="old"):
    # changer et inverser pour avoir df_anno comme fichier de base
    df_lm = pd.read_csv(csv_path)
    # dic avec les différents noms de colonnes de golden gender selon la version souhaitée (old, fem, nofem)
    golden_gender_cat = {"old":"old_golden_gender", "fem":"golden_gender_feminisation","nofem":"golden_gender_nonfem",
                         "new_fem": "label_fem", "new_nofem":"label_nofem"}

    # extraire annotations manuelles des fichiers dédiés et les ajouter automatiquement
    sampling_manu_files = {#"bloom-560m":"annotation_manuelle/sampling_lm_manu_annotated_gender_bloom560m.csv",
                            "bloom-560m" : "annotation_manuelle/bloom-560m_min.csv",
                           #"bloom-3b":"annotation_manuelle/annotation_manuelle_lm_sampling_3b.csv",
                            "bloom-3b": "annotation_manuelle/bloom-3b_min.csv",
                           "golden_sel": "annotation_manuelle/golden_selection.csv"}
    if "greedy" in csv_path:
        df_anno = pd.read_csv("annotation_manuelle/greedy_lm_manu_annotated_gender_bloom560m.csv")
    else:
        df_anno = pd.read_csv(sampling_manu_files[modele])

    lm = df_lm["output"]
    prompt = df_lm["prompt"]

    total_gender_theme = {}
    total_gender = [] #list with all identified gender in order to add to pd
    total_counter = [] #list with all counters to get dic with identified genders details for each letter
    total_markers = [] #same for gendered words that led to this gender identification

    for i,lettre in enumerate(lm):
        # garder la dernière phrase (incomplète) du prompt, qui finit par "car" MAIS pb filtre incomplet
        prompt2 = ".".join(prompt[i].split(".")[:-1])
        phrase_prompt = prompt[i].split(".")[-1]
        lettre_noprompt = lettre.split(prompt[i])[-1] #vraiment que la génération
        lettre = lettre.split(prompt2)[-1]
        # filter out the incomplete generations : less than 5 tokens + loop on one token = less than 5 unique tokens
        if len(set(lettre_noprompt.split())) > 5 and ("je" in lettre_noprompt or "j'" in lettre_noprompt or "Je" in lettre_noprompt or "J'" in lettre_noprompt):
            if version=="v7b":
                gender = get_gender_v7b(lettre)

        else:
            gender = ["incomplet/pas de P1",0,"none"]

        total_gender.append(gender[0])
        total_counter.append(gender[1])
        total_markers.append(gender[2])
        theme = df_lm["Theme"][i]

        if theme not in total_gender_theme:
            total_gender_theme[theme]=[]
        total_gender_theme[theme].append(gender[0])

    #lm2 = lm.assign(Identified_gender=total_gender)
    #lm_gender = pd.DataFrame(lm)
    df_lm["Identified_gender"]=total_gender

    df_lm["golden_gender"] = list(df_anno[golden_gender_cat[golden_gender_header]])
    df_lm["Detailed_counter"] = total_counter
    df_lm["Detailed_markers"] = total_markers

    path = csv_path.split("/")[1]

    #with open("gender_annotation_auto/"+path.split(".")[0]+f"_gender_print_{version}_md_tout.txt", "a") as f:
        #for theme, list_gender in total_gender_theme.items():
            #print(theme, ":", Counter(list_gender).most_common(), file=f)

    #with open("gender_annotation_auto/" + path.split(".")[0] + f"_gender_{version}_md_tout.json", "w") as f:
        #json.dump(total_gender_theme,f ,indent=4)

    prefix = "coverletter_sampling_"

    df_lm.to_csv("gender_output/"+prefix+path.split(".")[0]+f"_gender_{version}_{golden_gender_header}_trf.csv")


def detecter_genre(version,csv_path):
    """Processus de détection de genre et ajout des résultats dans des nouveaux CSV"""

    df_lm = pd.read_csv(csv_path)

    lm = df_lm["output"]
    lm.fillna("", inplace=True)
    prompt = df_lm["prompt"]

    total_gender_theme = {}
    total_gender = [] #list with all identified gender in order to add to pd
    total_counter = [] #list with all counters to get dic with identified genders details for each letter
    total_markers = [] #same for gendered words that led to this gender identification

    for i,lettre in enumerate(lm):
        # garder la dernière phrase (incomplète) du prompt, qui finit par "car" MAIS pb filtre incomplet
        #print(i,lettre)
        prompt2 = ".".join(prompt[i].split(".")[:-1])
        phrase_prompt = prompt[i].split(".")[-1]
        lettre_noprompt = lettre.split(prompt[i])[-1] #vraiment que la génération
        lettre = lettre.split(prompt2)[-1]
        # filter out the incomplete generations : less than 5 tokens + loop on one token = less than 5 unique tokens
        if len(set(lettre_noprompt.split())) > 5 and ("je" in lettre_noprompt or "j'" in lettre_noprompt or "Je" in lettre_noprompt or "J'" in lettre_noprompt):
            if version=="v7b":
                gender = get_gender_v7b(lettre)
        else:
            gender = ["incomplet/pas de P1",0,"none"]

        total_gender.append(gender[0])
        total_counter.append(gender[1])
        total_markers.append(gender[2])
        theme = df_lm["Theme"][i]  #"Theme"

        if theme not in total_gender_theme:
            total_gender_theme[theme]=[]
        total_gender_theme[theme].append(gender[0])

    #lm2 = lm.assign(Identified_gender=total_gender)
    #lm_gender = pd.DataFrame(lm)
    df_lm["Identified_gender"]=total_gender
    df_lm["Detailed_counter"] = total_counter
    df_lm["Detailed_markers"] = total_markers

    path = csv_path.split("/")[1]

    prefix = "coverletter_detection_v2_"

    #df_lm.to_csv("gender_output/"+prefix+path.split(".")[0]+f"_gender_{version}_trf.csv")
    df_lm.to_csv("gender_output/" + prefix + path.split(".")[0] + f"_gender_{version}_trf.csv")

# Je suis une personne de confiance, je suis très organisée et très proactive. => problème qui reste pcq head mal étiquetée

#t = "Ma formation en informatique me permet d'être polyvalent dans mes missions. Mes cheveux sont beaux. Ces compétences me permettent d'être gentil. Mon expérience professionnelle me permet d'être un bon gestionnaire de projets."

#test de Nicolas
"""t = "Nous rapportons le cas d'un patient âgé de 40 ans et gentille, sans antécédents pathologiques notables notamment pas d'asthme sous traitement médical ou chirurgical et qui n'a jamais bénéficié d'une appendicectomie en raison d'une hématurie chronique depuis l'âge de 16 ans avec des signes cliniques similaires à ceux décrits ci-dessus (douleur abdominale évoluant vers la fin du cycle menstruel associée à une altération de l'état général). L'examen avait objectivé dans ses deux premières consultations au Centre Hospitalier Universitaire Moulay Ismail Marrakech pour douleurs intestinales modérées intermittentes paroxystiques auxquelles s'ajoutaient des vomissements occasionnels ainsi qu'une fièvre épisodique remontant jusqu'à 39°C accompagnés d'autres signes généraux non spécifiques tels que faiblesse musculaire généralisée réduisant sa capacité physique quotidienne passablement limitée; il était stable sur le plan hémodynamique mais son état respiratoire restait instable fluctuant entre mauvais et très mauvais nécessitant plusieurs hospitalisations successives durant ces dernières années surtout quand on lui prescrivait certaines thérapeutiques antiinflammatoires stéroïdiens comme corticoïdes etc... Ces séances étaient relativement fréquentes allant même jusqu'à trois mois consécutifs chez certains patients atteint(e)s plus graves ayant eu besoin immédiat après leur apparition soit initialement pendant quelques semaines puis progressivement moins longtemps selon leurs progrès respectifs caractérisés avant tout par améliorations régulières si ce n'est spectaculaires tant quantitativement que qualitativement constatées essentiellement grâce à divers examens complémentaires biologiques effectués périodiquement telles que radiographies pulmonaires réalisées tous les six mois afin de détecter rapidement toute pneumopathie infectieuse résultant directement parfois indirectement encore indéniablement malgré toutes les précautions possibles prises soigneusement contre cette complication potentiellement mortelle devant laquelle nous avons toujours été confrontés face à notre insuffisance matérielle financière souvent inadéquate vis-à-vis nos ressources humaines limitées aussi bien morales que logistiques ne permettant évidemment aucune efficacité optimale dont témoigneraient déjà suffisamment diverses études scientifiques portant principalement sur différents aspects pertinents concernant cet aspect phénoménal complexe rendant difficile voire impossible appréciation exacte précise sinon systématiquement rigoureuse compte tenu également parmi autres facteurs expliquant peut-être largement partiels plutôt aléatoires donc incertains susceptibles néanmoins justifiant présomption réaliste faisant autorité dès lors provisoirement alors que seuls peuvent être retenus implicitement irréléctionnellement théoriquement vraisemblables hypothèses plausibles admettant nécessairement quelquefois probabilités minimes certes peu probables probablement exagérées impossibles assurément fausses affirmatives définitives convaincantes concluantes cohérentes satisfaisantes légitimatrices valides explicites compréhensibles accessibles utilisables exploitables applicables utiles intéressantes opportunes judicieuses adéquates diagnostiques curatives opératoires symptomatiques palliatifs conservateurs sublétaux temporaires transitoires ponctuels imminents relatifs futurs permanents durables ultérieurs perpétuels terminaux exceptionnels extrêmement improbables rarement rencontrés fort heureusement suprêmes lointains apocryphes imaginaires chimériques fantastiques hallucinogènes aberrants idéalistes utopiques philosophiques métaphysiques eschatologiques dogmatiques confessionnels sectaires fanatiques radicaux primitifs archaïques barbares outranciers monstrueux absurdes ridicules grotesques comiques burlesques extravagants farfelus capric"
print(get_gender_v7b(t, False))
exit()

#chercher pour "tout"
t1 = "Je pense correspondre à votre offre car  je suis une personne calme et sérieuse, j'aime le contact humain et je sais travailler en équipe. J'ai déjà travaillé comme aide aux cours, assistante administrative et je suis actuellement en formation dans un cabinet de recrutement. Je suis également une personne très organisée, j'aime avoir des dossiers à jour, j'aime bien faire des recherches sur internet. Enfin, je suis quelqu'un de très ponctuelle et je fais toujours de mon mieux pour terminer mon travail à temps. Je me présente : je m'appelle Aurore, j'ai 23 ans, je suis en formation dans un cabinet de recrutement et je suis actuellement étudiante en littérature appliquée à la documentation, communication, lettres et enseignement. Mon profil me correspond donc à 100% à votre offre. J'ai déjà travaillé comme aide aux cours, assistante administrative et je suis actuellement en formation dans un cabinet de recrutement. Je suis également une personne très organisée, j'aime avoir des dossiers à jour, j'aime bien faire des recherches sur internet. Enfin, je suis quelqu'un de très ponctuelle et je fais toujours de"
t2 = "Je pense correspondre à votre offre car  j'ai une grande curiosité pour les langues étrangères et les nouvelles cultures. Je suis une étudiante de 19 ans et j'adore le travail en équipe. Je suis très motivée à travailler pour une société internationale. J'habite à Madrid et j'ai des connaissances linguistiques très avancées. Je suis actuellement à la recherche d'un travail pour lequel je pourrais apprendre l'espagnol."
t3 = "Je pense correspondre à votre offre car  je peux faire du télétravail, avoir une certaine autonomie et je suis capable de gérer une équipe de travail. Je suis diplômée du Master Recherche en Histoire et Philosophie de l’Art Contemporain et je travaille à l’Université de Lorraine comme professeur de philosophie et d’histoire de l’art. J’interviens notamment dans des cours de spécialité (histoire des arts, art contemporain, histoire et philosophie de l’art) et en cours d’année, sur des problématiques de culture artistique. Je recherche un emploi à temps plein ou à temps partiel en tant que professeure des écoles. J’adore enseigner aux enfants et j’y prends beaucoup de plaisir. Je suis dynamique et sérieuse. Je suis professeur en lycée professionnel à Metz depuis plusieurs années et cherche à poursuivre une carrière en tant qu’enseignant. Je me propose donc de donner des cours de maths, français, philosophie ou anglais."
for t in [t1, t2, t3]:
    print(get_gender_v7b(t, False))
exit()"""

#detecter_genre("v7b", "annotation_manuelle/golden_selection.csv")
#exit()

#for modele in tqdm(["bloom-3b", "bloom-560m", "gpt2-fr","xglm-2.9B"]):
"""for modele in tqdm(["bloom-3b", "bloom-560m", "gpt2-fr","xglm-2.9B", "bloom-7b","vigogne-2-7b"]):
    print(modele)
    detecter_genre("v7b",f"expe_genre_output/genreno10_coverletter_sampling_{modele}.csv")"""

#for modele in ["bloom-3b", "bloom-7b", "bloom-560m", "gpt2-fr", "vigogne-2-7b", "xglm-2.9B"]:
for modele in tqdm(["gpt2-fr", "vigogne-2-7b", "xglm-2.9B"]):
    print(modele)
    detecter_genre("v7b",f"output/coverletter_sampling_{modele}.csv")