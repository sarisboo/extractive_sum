from nltk.tokenize import word_tokenize
import pandas as pd


def tokenizer(df, column):
    df[column].dropna(inplace=True)
    df["tokens"] = df[column].apply(word_tokenize)
    return df


if __name__ == "__main__":

    # Read the data
    train = pd.read_csv(
        "data/interim/segmented/train_clean_segmented.csv.gz", compression="gzip"
    )
    test = pd.read_csv(
        "data/interim/segmented/test_clean_segmented.csv.gz", compression="gzip"
    )

    val = pd.read_csv(
        "data/interim/segmented/val_clean_segmented.csv.gz", compression="gzip"
    )

    # Tokenize words
    train_tokenized = tokenizer(train, "sentence")
    test_tokenized = tokenizer(test, "sentence")
    val_tokenized = tokenizer(val, "sentence")

    print(train_tokenized.head())

    # Save data
    # Save as pandas .csv.gz without index
    train_tokenized.to_csv(
        "data/interim/filtered/train_tokenized.csv.gz",
        compression="gzip",
        index=False,
    )
    test_tokenized.to_csv(
        "data/interim/filtered/test_tokenized.csv.gz",
        compression="gzip",
        index=False,
    )
    val_tokenized.to_csv(
        "data/interim/filtered/val_tokenized.csv.gz",
        compression="gzip",
        index=False,
    )
