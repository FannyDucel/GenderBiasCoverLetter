"""Évaluer la détection automatique de genre :
1. Compter dans les annotations manuelles (pour voir ce que donnent les vrais résultats)
2. Comparer/évaluer la détection auto vs manuel
3. Essayer d'améliorer les scores de la détection auto : ajout de filtres ou essai de classif"""
from datetime import datetime
from typing import Dict, List
import re
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from tabulate import tabulate
from sklearn.metrics import classification_report

def multiple_replace(string, rep_dict):
    pattern = re.compile("|".join([re.escape(k) for k in sorted(rep_dict,key=len,reverse=True)]), flags=re.DOTALL)
    return pattern.sub(lambda x: rep_dict[x.group(0)], string)

def common_prefix(list_of_strings: List[str]) -> str:
    # Start with first character of first string and keep going
    if len(list_of_strings) == 1:
        return list_of_strings[0]

    prefix = ''
    for i in range(len(list_of_strings[0])):
        new_prefix = list_of_strings[0][:i]
        cond = [string.startswith(new_prefix) for string in list_of_strings]
        if all(cond):
            prefix = new_prefix
        else:
            return prefix

    return prefix


def common_suffix(list_of_strings: List[str]) -> str:
    list_of_strings = [x[::-1] for x in list_of_strings]
    return common_prefix(list_of_strings)[::-1]

def count_gender_by_field(gender, path_csv):
    df = pd.read_csv(path_csv)
    # print(df['golden_gender'].value_counts()) #counter total
    gendered_df = df[df["golden_gender"] == gender]
    count_by_theme = gendered_df.groupby("Theme").size()
    return count_by_theme

def counter_gender_by_field_to_csv(mode):
    for gender_category in ["Fem", "Masc", "Ambigu", "Neutre"]:
        if mode == "greedy":
            #path = "gender_output/coverletter_test_greedy_echantillon_2config_bloom_genderjeid_vsmanu.csv"
            path = "gender_output/coverletter_test_greedy_echantillon_2config_bloom_gender_v4_md.csv"
        else:
            #path = "gender_output/coverletter_sampling_echantillon_2config_bloom_fixed_genderjeid_vsmanu.csv"
            path = "gender_output/coverletter_test_sampling_echantillon_2config_bloom_fixed_gender_v4_md.csv"
        res = count_gender_by_field(gender_category,path)
        #print(gender_category, res, "\n*****************\n")#,file=f
        res.to_csv(f"gender_annotation/counter_manual_{mode}_{gender_category}_v4_tout.csv")

def compare_auto_manual(mode):
    """Donner nombre de bonnes et mauvaises identifications"""

    folder_dir = Path('./gender_output')
    assert folder_dir.exists()

    # A bunch of replacements to make the filenames make sense
    repls = {"test_sampling": "", "coverletter": "", "sampling": "", "tout_gender": "", "fixed_gender": "", "__": ""}

    dfs: Dict[str, pd.DataFrame] = {}
    for filename in folder_dir.rglob(f'coverletter*{mode}*csv'):
        key = multiple_replace(filename.name, repls)
        dfs[key] = pd.read_csv(filename)

    fname_prefix, fname_suffix = common_prefix(list(dfs.keys())), common_suffix(list(dfs.keys()))
    table = [["Version", "True", "False", "% True", "% False"]]

    for filename in sorted(dfs):
        df = dfs[filename]
        if "golden" in filename:
            df = df[df['Identified_gender'] != "incomplet/pas de P1"]
        else:
            df = df[df['Identified_gender'] != "incomplet"]
        n_annote = df.golden_gender.count()  #
        df_compare = df['golden_gender'][:n_annote] == df['Identified_gender'][:n_annote]

        if len(df_compare.value_counts()) > 1:
            total_counts = df_compare.value_counts().sort_index()  # sort_index pour que ce soit toujours dans le même ordre, pas croissant
            true_count = total_counts[1]
            false_count = total_counts[0]
            true_percent = true_count / sum(total_counts) * 100
            false_percent = false_count / sum(total_counts) * 100

            # Format filename
            filename = filename[len(fname_prefix):-len(fname_suffix)]

            table.append([filename, true_count, false_count, true_percent, false_percent])

    print("---TOTAL---")
    print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

def compare_auto_manual_details(mode):
    """Donner détails par genre, nombre de True/False par genre"""

    folder_dir = Path('./gender_output')
    assert folder_dir.exists()

    # A bunch of replacements to make the filenames make sense
    repls = {"test_sampling": "", "coverletter": "", "sampling": "", "tout_gender": "", "fixed_gender": "", "__": ""}

    dfs: Dict[str, pd.DataFrame] = {}
    for filename in folder_dir.rglob(f'coverletter*{mode}*csv'):
        key = multiple_replace(filename.name, repls)
        dfs[key] = pd.read_csv(filename)
        # print(filename)

    fname_prefix, fname_suffix = common_prefix(list(dfs.keys())), common_suffix(list(dfs.keys()))


    for gender in ['Masc', 'Fem', 'Neutre', 'Ambigu']:
        table = [["Version", "True", "False", "% True", "% False"]]
        for filename in sorted(dfs):
            df = dfs[filename]
            if "golden" in filename or "v7b" in filename:
                df = df[df['Identified_gender'] != "incomplet/pas de P1"]
            df = df[df['Identified_gender']!="incomplet"]
            n_annote = df.golden_gender.count() #nb of rows with a value in golden_gender
            df_gender = df.loc[df['golden_gender'] == gender]
            df_compare = df_gender['golden_gender'][:n_annote] == df_gender['Identified_gender'][:n_annote]
            filename = filename[len(fname_prefix):-len(fname_suffix)]

            if len(df_compare.value_counts()) > 1:
                total_counts = df_compare.value_counts().sort_index() #sort_index pour que ce soit toujours dans le même ordre, pas croissant
                true_count = total_counts[1]
                false_count = total_counts[0]
                true_percent = true_count / sum(total_counts) * 100
                false_percent = false_count / sum(total_counts) * 100
                table.append([filename, true_count, false_count, true_percent, false_percent])

            elif df_compare.empty:
                # print("Pas d'exemple pour le genre",gender)
                table.append([filename,  0,0,0,0])

            else:
                total_counts = df_compare.value_counts().sort_index()
                if str(total_counts)[
                    0] == "F":  # si la seule valeur est "false", remplir True avec 0 et inversement (fait à la sauvage parce que je trouve pas mieux)
                    false_count = total_counts[0]
                    true_count = 0
                else:
                    true_count = total_counts[0]
                    false_count = 0
                true_percent = true_count / sum(total_counts) * 100
                false_percent = false_count / sum(total_counts) * 100
                table.append([filename,  true_count, false_count, true_percent, false_percent])

        print("---", gender, "---")
        print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

def prec_recall_fscore(mode,versionlabel,modele="bloom-560m",trf=False):
    """Use sklearn to get classification report and overall precision, recall and fscore"""
    if trf:
        suf = "_trf"
    else:
        suf=""

    file_greedy = {"bloom-560m":"gender_output/coverletter_test_greedy_echantillon_2config_bloom_gender_v6_md_complet.csv"}
    file_sampling = {#"bloom-560m":"gender_output/coverletter_test_sampling_echantillon_2config_bloom_tout_gender_v6_nofem_complet_je.csv",
                     "bloom-560m":f"gender_output/coverletter_sampling_bloom-560m_min_gender_{versionlabel}{suf}.csv",
                     "bloom-3b":f"gender_output/coverletter_sampling_bloom-3b_min_gender_{versionlabel}{suf}.csv",
                     #"golden_sel":"annotation_manuelle/coverletter_sampling_golden_selection.csv",
                    "golden_sel":f"gender_output/coverletter_sampling_golden_selection_gender_{versionlabel}{suf}.csv",
                    "golden_test": "gender_output/coverletter_detection_fr-it_golden_selection_gender_v7b_trf.csv",
        "gold_test": "coverletter_detection_goldtestgolden_selection_gender_trf.csv",
        "epi+agtcorr+noun":"gender_output/coverletter_detection_epi+agtcorr+noun_golden_selection_gender_v7b_trf.csv",
        "agtcorr_b": "gender_output/coverletter_detection_agtcorr_bloom-560m_min_gender_v7b_trf.csv",
    "golden_test_nocorr_noepi":"gender_output/coverletter_detection_fr-it_nocorr_noepi_golden_selection_gender_v7b_trf.csv"}


    if mode == "greedy":
        df = pd.read_csv(file_greedy[modele])
    else:
        df = pd.read_csv(file_sampling[modele])

    df = df[df['Identified_gender'] != "incomplet/pas de P1"]
    #df = df[df['comments'] != "Hors-sujet"]
    #n_annote = df.golden_gender.count()
    n_annote = df.label_fem.count()
    #y_true = df["golden_gender"].loc[:n_annote].to_numpy()
    y_true = df["label_fem"].loc[:n_annote].to_numpy()
    #y_true = np.array(df["golden_gender"].loc[:n_annote])
    y_pred = df["Identified_gender"].loc[:n_annote].to_numpy()

    prec, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    #labels = list(set(y_true))

    with open(f"scores_metrics/classification_report_{mode}_{versionlabel}_{n_annote}_{modele}{suf}.txt", "w") as f:
        print(datetime.now(), file=f)
        print(classification_report(y_true, y_pred, digits=4), file=f) #target_names=labels,

    return prec, recall, fscore, support


def prec_recall_fscore_it(modele):
    """Use sklearn to get classification report and overall precision, recall and fscore"""

    file_sampling = {"it_papa": "ressources_it/it_coverletter_detection_annotation-italien-papa_gender_lg.csv",
                     "it_xheni": "ressources_it/it_coverletter_detection_annotation-italien-xheni_gender_lg.csv",
                     "it_combi": "ressources_it/it_coverletter_detection_annotation_gender_lg.csv",
                     "it_siy": "ressources_it/it_coverletter_detection_it_annotation_siyana_gender_lg.csv",
                     "it_combi_no90": "ressources_it/it_coverletter_detection_annotation_no90_gender_lg.csv",
                     "it_combi_no20": "ressources_it/it_coverletter_detection_annotation_no20_gender_lg.csv"}
    df = pd.read_csv(file_sampling[modele])
    #df['majority_gender'] = df['majority_gender'].map(
        #{'Mas': 'Masc', 'Neutral': 'Neutre', "Fem": "Fem", "Ambiguous": "Ambigu", "Neutral?": "Neutre"})

    df = df[df['Identified_gender'] != "incomplet"]
    #df = df[df['comments'] != "Hors-sujet"]
    #n_annote = df.golden_gender.count()
    n_annote = df.genre.count()
    #y_true = df["golden_gender"].loc[:n_annote].to_numpy()
    y_true = df["genre"].loc[:n_annote].to_numpy()
    #y_true = np.array(df["golden_gender"].loc[:n_annote])
    y_pred = df["Identified_gender"].loc[:n_annote].to_numpy()

    prec, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')
    #labels = list(set(y_true))

    with open(f"scores_metrics/classification_report_{n_annote}_{modele}.txt", "w") as f:
        print(datetime.now(), file=f)
        print(classification_report(y_true, y_pred, digits=4), file=f) #target_names=labels,

    return prec, recall, fscore, support

#print(compare_auto_manual_details_old("greedy"))
#print(vp_fp("sampling"))
#print(prec_recall_fscore("sampling","v7b_new_nofem", "golden_sel"))

#print(prec_recall_fscore_it("it_siy"))
#print(prec_recall_fscore_it("it_combi"))
print(prec_recall_fscore_it("it_combi_no90"))


exit()

print(prec_recall_fscore("sampling","v7b_new_fem", "epi+agtcorr+noun", False))
exit()
print(prec_recall_fscore("sampling","v7b_new_fem", "golden_sel", False))
exit()
print(prec_recall_fscore("sampling","v7b_new_nofem", "golden_sel", True))


print(prec_recall_fscore("sampling","v7b_fem", "bloom-560m", True))
print(prec_recall_fscore("sampling","v7b_nofem", "bloom-560m", True))

print(prec_recall_fscore("sampling","v7b_fem", "bloom-3b", True))
print(prec_recall_fscore("sampling","v7b_nofem", "bloom-3b", True))

#print(prec_recall_fscore("sampling", "bloom-3b"))
exit()

print("*******************SAMPLING*******************")
compare_auto_manual("sampling")
compare_auto_manual_details("sampling")
#print(compare_auto_manual_details("sampling"))
