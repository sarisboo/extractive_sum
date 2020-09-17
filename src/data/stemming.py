from nltk.stem import PorterStemmer
from ast import literal_eval
import pandas as pd

# Porter Stemmer is the oldest, it uses suffix stripping.
# It uses rules to decide whether it is wise to strip a suffix.
# we use it for its simplicity and speed
porter = PorterStemmer()


def stemmer(stemmed_sent):
    """
    Removes stop words from a tokenized sentence
    """
    porter = PorterStemmer()
    stemmed_sentence = []
    for word in literal_eval(stemmed_sent):
        stemmed_word = porter.stem(word)
        stemmed_sentence.append(stemmed_word)
    return stemmed_sentence


if __name__ == "__main__":

    # Load Data
    train = pd.read_csv(
        "data/interim/no_stop_words/train_no_stop_words.csv.gz", compression="gzip"
    )
    test = pd.read_csv(
        "data/interim/no_stop_words/test_no_stop_words.csv.gz", compression="gzip"
    )
    val = pd.read_csv(
        "data/interim/no_stop_words/val_no_stop_words.csv.gz", compression="gzip"
    )

    # Drop NAs
    train["no_stop_words_tokens"].dropna(inplace=True)
    test["no_stop_words_tokens"].dropna(inplace=True)
    val["no_stop_words_tokens"].dropna(inplace=True)

    # Remove Stop Words
    train["stemmed_tokens"] = train["no_stop_words_tokens"].apply(lambda x: stemmer(x))
    test["stemmed_tokens"] = test["no_stop_words_tokens"].apply(lambda x: stemmer(x))
    val["stemmed_tokens"] = val["no_stop_words_tokens"].apply(lambda x: stemmer(x))

    print(train["stemmed_tokens"].head())
    # Save to csv

    train.to_csv(
        "data/interim/stemmed/train_stemmed.csv.gz",
        compression="gzip",
        index=False,
    )

    test.to_csv(
        "data/interim/stemmed/test_stemmed.csv.gz",
        compression="gzip",
        index=False,
    )

    val.to_csv(
        "data/interim/stemmed/val_stemmed.csv.gz",
        compression="gzip",
        index=False,
    )
