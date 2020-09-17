import pandas as pd
from nltk.corpus import stopwords
from ast import literal_eval


def stop_words_remover(tokenized_sent):
    """
    Removes stop words from a tokenized sentence
    """
    # Convert string back to list

    filtered_sentence = []
    stop_words = set(stopwords.words("english"))
    for word in literal_eval(tokenized_sent):
        if word not in stop_words:
            filtered_sentence.append(word)
    return filtered_sentence


if __name__ == "__main__":

    # Load Data
    train = pd.read_csv(
        "data/interim/filtered/train_tokenized.csv.gz", compression="gzip"
    )
    test = pd.read_csv(
        "data/interim/filtered/test_tokenized.csv.gz", compression="gzip"
    )
    val = pd.read_csv("data/interim/filtered/val_tokenized.csv.gz", compression="gzip")

    # Drop NAs
    train["tokens"].dropna(inplace=True)
    test["tokens"].dropna(inplace=True)
    val["tokens"].dropna(inplace=True)

    # Remove Stop Words
    train["no_stop_words_tokens"] = train["tokens"].apply(
        lambda x: stop_words_remover(x)
    )
    test["no_stop_words_tokens"] = test["tokens"].apply(lambda x: stop_words_remover(x))
    val["no_stop_words_tokens"] = val["tokens"].apply(lambda x: stop_words_remover(x))

    # Save to csv

    train.to_csv(
        "data/interim/no_stop_words/train_no_stop_words.csv.gz",
        compression="gzip",
        index=False,
    )

    test.to_csv(
        "data/interim/no_stop_words/test_no_stop_words.csv.gz",
        compression="gzip",
        index=False,
    )

    val.to_csv(
        "data/interim/no_stop_words/val_no_stop_words.csv.gz",
        compression="gzip",
        index=False,
    )
