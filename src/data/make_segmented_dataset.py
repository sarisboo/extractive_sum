import pandas as pd
import re

# Creating `text_id `column from index
def make_text_id(df):
    df["text_id"] = df.index
    df = df[["text_id", "article", "highlights"]]
    return df


def split_into_2_dfs(df):
    df_1 = df[["text_id", "article"]]
    df_2 = df[["text_id", "highlights"]]

    return df_1, df_2


def split_sentences(text):
    # Segment texts into sentences
    r_sentence_boundary = re.compile(
        r"\s?[.!?]\s?"
    )  # Modify this to not include abbreviations and other exceptions
    return r_sentence_boundary.split(text)[:-1]


# Split text by sentences
def split_by_sentence(df):
    df["sentences"] = df["article"].apply(lambda x: split_sentences(str(x)))


# Make a list of (text_id, sentence_list) pairs
def tup_list_maker(tup_list):
    """
    Takes a list of tuples with index 0 being the text_id and index 1 being a
    list of sentences and broadcasts the text_id to each sentence
    """
    final_list = []
    for item in tup_list:
        index = item[0]
        sentences = item[1]
        for sentence in sentences:
            pair = (index, sentence)
            final_list.append(pair)
    return final_list


# Create a list of tuples containing in index0, text_id and in index 1 the list of sentences corresponding to this text
def create_full_tuple(df):
    tuples = list(zip(df["text_id"], [sentence for sentence in df["sentences"]]))
    tup_list = tup_list_maker(tuples)
    # Converting the tuples list into a dataframe
    sentences = pd.DataFrame(tup_list, columns=["text_id", "sentence"])
    return sentences


# Create full dataframe


def create_full_final_dataframe(df):

    """
    Creates the final segmented dataframe with the `is_summary` column
    """

    dataframe = make_text_id(df)
    df_article, df_highlights = split_into_2_dfs(dataframe)

    df_article["sentences"] = df_article["article"].apply(
        lambda x: split_sentences(str(x))
    )
    df_highlights["sentences"] = df_highlights["highlights"].apply(
        lambda x: split_sentences(str(x))
    )
    segmented_df_articles = create_full_tuple(df_article)
    segmented_df_highlights = create_full_tuple(df_highlights)

    # Create targets for dataframes
    segmented_df_articles["is_summary_sentence"] = 0
    segmented_df_highlights["is_summary_sentence"] = 1

    # Stack the 2 dataframes and order by `text_id` column
    return segmented_df_articles.append(
        segmented_df_highlights, ignore_index=True
    ).sort_values(by=["text_id"])


if __name__ == "__main__":

    # Load data
    train = pd.read_csv("data/interim/cnn_dm_train.csv.gz", compression="gzip")
    test = pd.read_csv("data/interim/cnn_dm_test.csv.gz", compression="gzip")
    val = pd.read_csv("data/interim/cnn_dm_val.csv.gz", compression="gzip")

    # Select only needed columns
    train = train[["article", "highlights"]]
    test = test[["article", "highlights"]]
    val = val[["article", "highlights"]]

    # Segmenting datasets
    train_clean_segmented = create_full_final_dataframe(train)
    test_clean_segmented = create_full_final_dataframe(test)
    val_clean_segmented = create_full_final_dataframe(val)

    # Save as pandas .csv.gz without index
    train_clean_segmented.to_csv(
        "data/interim/segmented/train_clean_segmented.csv.gz",
        compression="gzip",
        index=False,
    )
    test_clean_segmented.to_csv(
        "data/interim/segmented/test_clean_segmented.csv.gz",
        compression="gzip",
        index=False,
    )
    val_clean_segmented.to_csv(
        "data/interim/segmented/val_clean_segmented.csv.gz",
        compression="gzip",
        index=False,
    )
