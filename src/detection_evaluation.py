from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

def prec_recall_fscore(language):
    """Use sklearn to get classification report and overall precision, recall and fscore
    by comparing manual and automatic gender labels on the manually annotated generations."""
    file_path = f"annotated_texts/{language}/annotated-manual_annotation_{language}.csv"

    df = pd.read_csv(file_path)

    if language == "FR":
        df = df[df['Identified_gender'] != "incomplet/pas de P1"]
        n_annote = df.label_fem.count()
        y_true = df["label_fem"].loc[:n_annote].to_numpy()
    if language == "IT":
        df = df[df['Identified_gender'] != "incomplet"]
        n_annote = df.genre.count()
        y_true = df["genre"].loc[:n_annote].to_numpy()

    y_pred = df["Identified_gender"].loc[:n_annote].to_numpy()

    prec, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, average='macro')

    with open(f"results/{language}/detection_system_eval/classification_report_{language}.txt", "w") as f:
        print(datetime.now(), file=f)
        print(classification_report(y_true, y_pred, digits=4), file=f) #target_names=labels,

    return prec, recall, fscore, support


print(prec_recall_fscore("FR"))
print(prec_recall_fscore("IT"))