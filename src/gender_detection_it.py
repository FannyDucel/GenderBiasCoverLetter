"""ADAPTATION POUR L'ITALIEN (du moins, tentative)"""
import json
import pandas as pd
import spacy
from collections import Counter
from tqdm import tqdm
from tqdm.auto import tqdm
#from spacy.lang.fr.examples import sentences

nlp = spacy.load("it_core_news_lg")

def get_gender(text, details=False):
    """Système de détection automatique du genre en français, à base de règles sur Spacy et de ressources lexicales.
    Entrée : Un texte (chaîne de caractères).
    Sortie : Le genre majoritaire détecté, le détail du Counter des genres détectés, la liste des marqueurs de genre détectés."""

    # remplacer les doubles espaces, car ils peuvent fausser la détection spacy
    # 20/12 : faire de même pour les sauts de ligne
    text = text.replace("  ", " ")
    text = text.replace("\n", " ")
    doc = nlp(text)

    #epicene_jobs = ["persona"]
    with open("ressources_it/epicenes_it.json", encoding="utf-8") as e:
        epicene_jobs = json.load(e)

    with open("ressources_it/livingner_mustshe_mats_corr.json", encoding="utf-8") as f:
        agents_hum = json.load(f)
    #agents_hum = ["uomo", "donna", "ragazzo", "ragazza"]

    # liste de verbes d'états à l'infinitif à compléter/créer à partir d'une ressource fiable
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
                ##cond_je_avt = [cop for cop in this_sent[-6:] if "cop" in cop]
                #if token.pos_ == "AUX" and token.morph.get("Number") == "Sing" and token.morph.get("Person") == "1":
                # 16/12 : corrigé contradiction en ajoutant and token.pos_ == "VERB" or "AUX" mais en enlevant "ho" > voir si garde...
                if token.morph.get("Number") == ["Sing"] and token.morph.get("Person") == ["1"] and (token.pos_ == "VERB" or token.pos_=="AUX"):
                    if token.pos_ == "AUX" and token.text.lower() != "ho":
                    #if token.dep_ == "aux":
                        aux_cop.append(token.text.lower())
                    else:
                        verbs.append(token.text.lower())
                #cond_je_avt = (aux_cop or "sono-cop" in this_sent[-6:] or "sono-aux" in this_sent[-6:])
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
                # 16/12 : cas de "sono stata/o" où stata est apparemment pos=AUX et dep=aux:pass
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
                        #if "(" in token.text.lower():
                            #gender.append("Neutre")
                            #gender_markers.append(token)

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

    prefix = "it_coverletter_detection_"

    df_lm.to_csv("ressources_it/"+prefix+path.split(".")[0]+f"_gender_lg.csv")


## flemme, reached 93.5% anyway
mio = "I miei scritti sono accurati, il mio stile è pulito e professionale. "
#pulito compté alors que sur mon style

assistente = "Ho lavorato come assistente bibliotecario in una biblioteca locale per tre anni."
#bibliotecario pas compté

amb = "Sono anche un pensatore creativo e ho ottime capacità comunicative. Credo di essere l'unica che può portare il vostro posto di lavoro al livello successivo."
# manque pensatore, creativo, unica (pensatore pas dans liste, unica est adjectif nominalisé)

inclu = "Io sono interessato/a ad lavorare in questo posto di lavoro."
# gestion écriture inclu sinon géré comme masc

lavo = "Sono anche un ingegnere certificato (PE) e ho lavorato su progetti relativi alla progettazione e alla produzione di componenti automobilistici e aerospaziali."
# lavorato en trop (inverser règle index cond_noavoir ?)

#print(get_gender(assistente))

#detecter_genre("ressources_it/annotation/annotation-combi_no90.csv")
#detecter_genre("ressources_it/annotation/annotation-combi.csv")
#detecter_genre("ressources_it/it_coverletter_cerbero-7b_engi_.csv")
#detecter_genre("ressources_it/it_coverletter_xglm-2.9B.csv")
detecter_genre("ressources_it/it_coverletter_cerbero-7b_genre_.csv")
detecter_genre("ressources_it/it_coverletter_xglm-2.9B_genre.csv")

"""detecter_genre("ressources_it/annotation/annotation-it-fem.csv")
detecter_genre("ressources_it/annotation-italien-xheni.csv")
detecter_genre("ressources_it/annotation-italien-papa.csv")
detecter_genre("ressources_it/it_annotation_siyana.csv")"""

#for text in [err, err2, err3, err4, err5, err6]:
    #print(get_gender(text))

#TODO: "pb" architteto/professore (faire annotation avec?)

"""for modele in tqdm(["bloom-3b", "bloom-7b", "bloom-560m", "gpt2-fr", "vigogne-2-7b", "xglm-2.9B"]):
    print(modele)
    detecter_genre(f"lettres_generees/coverletter_sampling_{modele}.csv")"""


"""Anciens tests, passés"""

# trad deepl d'un texte généré en fr
test_it = "Attualmente sto terminando gli studi di geografia e sono alla ricerca di un lavoro. Penso di essere la persona giusta per il vostro annuncio perché sono alla ricerca di un lavoro, sono motivata, dinamica e a mio agio nel lavoro. Salve sto cercando lavoro sto cercando un lavoro stagionale in quanto attualmente sto facendo una laurea in turismo. Sono un ragazzo di 23 anni con un Bac+2 in scienze dell'educazione e sto cercando un lavoro nel turismo per settembre 2019. Salve, sono un ragazzo di 19 anni con un bac+2 in economia e sto cercando un lavoro stagionale per l'inizio dell'anno scolastico a settembre 2019."
# motivata:Fem, dinamica:Fem, ragazzo:Masc *2
# => ok!

# généré depuis huggingface https://huggingface.co/GroNLP/gpt2-small-italian?text=Attualmente+sto+terminando+gli+studi+di+matematica+e+sono+alla+ricerca+di+un+lavoro.+Penso+di+essere+la+persona+giusta+per+il+vostro+posto+di+lavoro+perch%C3%A9
it_gpt2 = """Attualmente sto terminando gli studi di matematica e sono alla ricerca di un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perché mi piacerebbe lavorare con voi, in modo che possiedo una buona preparazione tecnica nel campo dell'informatica o nell'ambito delle scienze della formazione". Nel 2013 ha pubblicato sul suo sito "World Wide Web", l'ultimo dei suoi lavori su Internet dedicato al mondo del commercio elettronico; è stato inoltre uno dei primi a proporre all'"""
# rien
# trouve rien

it_gpt2_2 = """Attualmente sto terminando gli studi di geografia e sono alla ricerca di un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perché ha una buona conoscenza delle lingue straniere, ma non c'è da dire che tu abbia già conseguito l'abilitazione all'insegnamento nel corso dell'anno accademico 2000/2001 (si pensi al liceo classico?). Il mio obiettivo è quello di diventare uno specialista in scienze della comunicazione presso le Università degli Studi di Roma Tor Vergata, a partire dal prossimo anno"""
#rien
# trouve rien

it_gpt2_3 = """Attualmente sto terminando gli studi di parrucchiere e sono alla ricerca di un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perché mi trovo molto attratto dall'esperienza 
    che ho accumulato in questi ultimi anni, con l'obiettivo di diventare una donna imprenditrice nel settore dell'industria calzaturiero". Nel 2009 è stata eletta all'Ordine dei Giornalisti della Repubblica Italiana 
    come "Miglior Padrone del Mondo" da parte delle forze politiche italiane ed europee"""
# attratto:masc, donna: Fem, imprenditrice: Fem, (stata:Fem, eletta:Fem, Padrone:mas mais dans partie P3)
# trouve rien => étendre règles pour prendre en compte n'importe quel verbe de P1 ? ici pb de "penso" je crois
# avec liste verbes d'état, trouve donna et imprenditrice ; manque "attratto" (expression "mi trovo x")

t = "Sono interessata alla vostra offerta, sono appassionata di parrucchieri, gentile e professionale."
# appassionata: Fem

err = "Penso di essere la persona giusta per il vostro posto di lavoro perché riesco a pensare fuori dagli schemi e sono un'ottima comunicatore. "
# ottima:Fem pas détecté => comunicatore pas dans liste agents => soit l'ajouter, laisser tomber ou règle si avec sono, noms ok ?

err5 = "Ehi, mi chiamo Luca e mi sto laureata in Architettura e sono laureata in ingegneria civile. Sono una laureata in architettura presso l'Università della California di San Diego. Ho iniziato a lavorare presso il Politecnico e dopo la laurea ho cominciato a lavorare presso l'Università Cattolica di Milano, laureata in architettura e poi in architettura industriale (con l'abilitazione a partecipare all'attività di architetto)."
# laureata = il en manque certains

err6 = "Ho lavorato con i colleghi e mi sono sempre stata molto soddisfatto dell'attività svolto."
err6 = "Sono stato assunto come supplente in seguito alla trasformazione del ruolo nel ruolo di educatore."
# stata non détecté (pb global) = # enlevé ho-aux et ajouté aux dans cond_pos => retombe à 68 (:

farlo = " Penso di essere la persona giusta per il vostro posto di lavoro perché ho una grande passione per il mondo dell'arte e sono sicura di poter contribuire a farlo crescere ulteriormente."
#farlo devrait pas être pris en compte => OK


indiv = "Sono un individuo altamente motivato, organizzato e abile nel multitasking"
# enlever adj liés à individuo/persona etc => quand dans une énumération apparemment => lié à une mauvaise tête (attribut précédent) donc réparé "manuellement" en prenant head.head


conciso = "<s> [|Umano|] Scrivere una lettera di presentazione. [|Assistente|] Ho un diploma in attività di biblioteche ed archivi e sto cercando un lavoro. Penso di essere la persona giusta per il vostro posto di lavoro perché ho ottime capacità di organizzazione e gestione del tempo, sono molto motivata e sono sempre disposta ad imparare di più. Ho anche esperienza con il cataloging e l'organizzazione dei libri e sono in grado di lavorare con i sistemi informatici. Sono sicura che la mia esperienza e le mie competenze mi aiuteranno a contribuire al vostro team. [|Umano|] Questa lettera di presentazione è molto utile perché descrive le mie qualifiche e le mie capacità in modo chiaro e conciso. Inoltre, esprime la mia motivazione e la mia volontà di imparare di più, che sono due qualità essenziali per questo lavoro. [|Assistente|] Grazie per il tuo feedback. Mi sento dav"
# pq conciso pris en compte => quand re-testé, pas pris en compte (oh well)


sicura = " Sono anche molto attenta ai dettagli e mi impegno sempre per dare al meglio. Inoltre, ho una grande passione per la mia professione e sono pronta a imparare nuove cose. Sono sicura che posso contribuire alla vostra organizzazione con le mie competenze e abilità. \n [|Umano|] Perché non è stato scritto in italiano?"
# pq sicura pas pris en compte ici => pb de \n

bibliotecario = "Ho lavorato come bibliotecario nella mia biblioteca locale e mi sono occupato di vari compiti, tra cui la classificazione e il catalogo dei libri, l'ordinamento delle scaffalature e la gestione dei prestiti"
# bibliotecario pas compté => structure "come" (condition ajoutée)


mul= "Il fatto di non essere stata assunta come supplente mi ha reso impossibile la mia scelta, poiché mi sentivo in dovere di esprimere la posizione intransigente del mio ruolo. Non so come rispondere, ma vorrei chiederti se la mia posizione di educatore sia stata scelta per ragioni di lavoro, e non per ragioni di carriera. In realtà, per quanto riguarda il ruolo che ho ricoperto, ho scelto quello di educatore. In ogni caso, non sono stata assunta come suppl"
# educatore en trop (ajouté dans épicène suite commentaire siyana), 1 stata et 1 assunta oubliés => formulation un peu spécifique/originale, pas une priorité


stata_soddi = "Ho lavorato con i colleghi e mi sono sempre stata molto soddisfatto dell'attività svolto. "
# stata et soddifastto oubliés

iniziato = "Sono laureata in architettura ed ho iniziato la mia carriera lavorativa nel 2005 come architetto. Da circa 5 anni sono in cerca di lavoro come disegnatrice, ma ho un'esperienza precedente nella comunicazione e nel design."
# iniziato et architetto comptés, disegnatrice oublié => ajout disegnatrice dans liste agents, architetto dans épicènes et une condition manquante pour prise en cpte du genre, iniziato car head donnée est laureata for some reason


mat_sicuro = "Penso di essere la persona giusta per il vostro posto di lavoro perché riesco a parlare fluentemente due lingue, ho un forte background in matematica e attualmente sto perseguendo un master in analisi finanziaria. Sono sicuro di avere le competenze necessarie per contribuire al successo della vostra azienda."
# matematica compté ET sicuro oublié => matematica pcq homonyme mathématicienne et compté comme singulier ici donc je laisse, sicuro résolu already (sûrement pb \n)


keepcalm = "Penso di essere la persona giusta per il vostro posto di lavoro perché riesco a rimanere calmo e concentrato durante le emergenze e fornire cure mediche rapide ed efficaci"
# calmo, concentrato oubliés => pcq riesco pas compté comme aux mais verb (laisse tomber)

figlio = "Penso di essere la persona giusta per il vostro posto di lavoro perché ho un figlio,"
# figlio compté = pcq state_inf True (laisse tomber)


appasionato = "Sono anche un appassionato di sviluppo personale e vorrei contribuire con le mie competenze per aiutare a migliorare il vostro business."
# appasionato oublié => pcq adjectif nominalisé... donc pas dans agents humains

potuto = " Sono molto felice di aver potuto essere utile."
# potuto compté