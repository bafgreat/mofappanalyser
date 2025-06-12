import pandas as pd
from mofappanalyser.read_write import filetyper


def combine_text(row):
    return " ".join([
        str(row.get("title", "")),
        str(row.get("keywords", "")),
        str(row.get("keywords-plus", "")),
        str(row.get("abstract", ""))
    ])


def clean_df(filename):
    """
    Clean the DataFrame by removing trailing commas in JSON files.
    """
    df = filetyper.load_data(filename)
    df["title"] = df["title"].combine_first(df["booktitle"])
    df.drop(columns=["booktitle"], errors="ignore")
    df_text = pd.DataFrame()
    df_text["unique-id"] = df["unique-id"]
    df_text["year"] = df["year"]
    df_text["n_citation"] = df["times-cited"]
    df_text["journal"] = df["journal"]
    df_text["doi"] = df["doi"]
    df_text["research-areas"] = df["research-areas"]
    df_text["funding-text"] = df["funding-text"]
    df_text["text"] = df.apply(combine_text, axis=1)

    df_text.to_csv('../../data/final_cleaned_mofs_articles_data.csv', index=False)

    # print(df_text["research-areas"].unique())


# clean_df('../../data/final_mof_articles_data.csv')