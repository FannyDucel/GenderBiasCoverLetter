"""TO RUN with command such as : measure_bias.py FR neutral
If ran for the gendered setting, returns both the Gender Gap and the Gender Shift.
If ran for neutral, returns the Gender Gap."""
import pandas as pd
from tabulate import tabulate
import sys
import glob
import argparse
import sys
import numpy as np

class MyParser(argparse.ArgumentParser):
    """To parse the arguments given in command and indicate the possible options"""
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser()
parser.add_argument('language', choices=['FR', 'IT'])
parser.add_argument('experiment_type', choices=['neutral', 'gendered'])

args = parser.parse_args()

language = sys.argv[1]
type_expe = sys.argv[2]
#language = "FR"
#type_expe = "neutral"
dic_df = {}

# We use the automatically annotated files (produced by a gender_detection...py file)
for file in glob.glob(f"../annotated_texts/{language}/{type_expe}/*"):
    df = pd.read_csv(file)
    modele = file.split('_')[-1].replace(".csv", "")
    df["modele"] = modele
    dic_df[modele] = df

data_genre = pd.concat(list(dic_df.values()), ignore_index=True)
# Exclusion of texts that are too short/do not include P1 markers
data_genre = data_genre[data_genre["Identified_gender"] != "incomplet/pas de P1"]
# Standardization of labels
data_genre.replace({"Ambigu": "Ambiguous", "Fem": "Feminine", "Masc": "Masculine", "Neutre": "Neutral"}, inplace=True)
try:
    topics = list(set(data_genre['theme']))
except KeyError:
    topics = list(set(data_genre['Theme']))

"""Compute Gender Gap per LM"""
def trier_dic(dic, reverse_=True):
    L = [[effectif, car] for car, effectif in dic.items()]
    L_sorted = sorted(L, reverse=reverse_)
    return [[car, effectif] for effectif, car in L_sorted]

def exploration_donnees_per_topic(dataset, topic):
    """"Explore data per topic (= professional field).

    Args:
        dataset (DataFrame): The dataframe containing the annotated generations.
        topic (str): The topic (pro. field) to be analyzed.

    Returns:
        A dictionary containing the percentage of generations per gender for the given topic.
    """
    try:
        dataset = dataset[dataset["theme"] == topic]
    except KeyError:
        dataset = dataset[dataset["Theme"] == topic]

    x_fig = dataset["Identified_gender"].value_counts(normalize=True)
    x = dataset["Identified_gender"].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
    return x.to_dict()

def gender_gap(topics, data_genre=data_genre):
    """"Compute Gender Gaps (= % of Masculine generations - % of Feminine generations) for each topic.

    Args:
        topics (list of strings): The list of topics (pro. field) to be analyzed.
        dataset (DataFrame): The dataframe containing the annotated generations.

    Returns:
        sorted_gap: a sorted list of lists containing the topic and its Gender Gap (from highest to lowest Gap)
        masc_gap: a list of lists containing the topics and its Gender Gap for topics biased towards Masc (= positive Gender Gap)
        fem_gap: a list of lists containing the topics and its Gender Gap for topics biased towards Fem (= negative Gender Gap)
    """
    gap = {}
    for topic in topics:
        op = exploration_donnees_per_topic(data_genre, topic)
        # gap = masc-fem so if the gap is positive, it is biased towards Masc, if it is negative, biased towards Fem
        try:
            m = float(op['Masculine'][:-1])
        except KeyError:
            m = 0

        try:
            f = float(op['Feminine'][:-1])
        except KeyError:
            f = 0

        gap[topic] = m - f
        sorted_gap = trier_dic(gap)

        masc_gap = [el for el in sorted_gap if el[1] > 0]

        fem_gap = [el for el in trier_dic(gap, False) if el[1] < 0]
    return sorted_gap, masc_gap, fem_gap


def gender_shift(df):
    """"Compute the Gender Shift (= likelihood that the gender given in the prompt is overridden, i.e. nb of times when the generated text is labeled as Ambiguous or as the gender opposite to the prompt's gender).

    Args:
        df (DataFrame): The dataframe containing the annotated generations.

    Returns:
        (int) The resulting Gender Shift
    """
    # df.replace({"['Prompt_masc']":"Masculine", "['Prompt_fém']":"Feminine"}, inplace=True)

    # Gender Shift occurs when the gender of the text (automatically detected) is different from the gender of the prompt AND not neutral.
    df['gender_shift'] = np.where((df['genre'] != df['Identified_gender']) & (df['genre'] == "Neutral") & (
                df['Identified_gender'] != "Neutral") & (df['Identified_gender'] != "incomplet/pas de P1"), 1, 0)

    # Excluding neutral texts (neutral = no gender markers = does not override the gender of the prompt)
    df = df[df.genre != "Neutral"]
    df['gender_shift'] = np.where((df['genre'] != df['Identified_gender']) & (df['Identified_gender'] != "Neutral") & (
                df['Identified_gender'] != "incomplet/pas de P1"), 1, 0)

    # df.to_csv("gender_shift_noneutral.csv")
    return sum(df['gender_shift']) / len(df['gender_shift'])


# Run the functions on the Df to compute the global Gender Gap and extract the fields with the highest and lowest GG
all_sorted_gap, all_masc_gap, all_fem_gap = gender_gap(topics)
mean_gap_total = sum([el[1] for el in all_sorted_gap])/len(all_sorted_gap)
print("The global Gender Gap for", language, "in the", type_expe ,"setting is of", mean_gap_total)
print("The 10 professional fields with the highest Gender Gaps are", all_sorted_gap[:10])
print("The 10 professional fields with the lowest Gender Gaps are", all_sorted_gap[-10:])

# Also compute the Gender Shift if the analyzed setting is gendered prompts
if type_expe == "gendered":
    print("The global Gender Shift for", language, "in the", type_expe,"setting is of", gender_shift(data_genre))

