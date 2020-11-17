import pandas as pd
import numpy as np
from ast import literal_eval

# evaluating rows to output proper data type
def lit_eval(df, col):
    df[col] = df[col].apply(lambda x: literal_eval(x))
    return df


# Remove 2 word or less 'Sentences':
def remove_no_sentence(sentence):
    if len(sentence) <= 2:
        sentence = np.nan
    else:
        sentence = sentence
        return sentence


if __name__ == "__main__":

    # Reading the Data
    train = pd.read_csv("data/interim/stemmed/train_stemmed.csv.gz", compression="gzip")
    test = pd.read_csv("data/interim/stemmed/test_stemmed.csv.gz", compression="gzip")
    val = pd.read_csv("data/interim/stemmed/val_stemmed.csv.gz", compression="gzip")

    # Drop Nas
    train.dropna(inplace=True)
    test.dropna(inplace=True)
    val.dropna(inplace=True)

    # Evaluating the string containing a Python literal (list)
    train_eval = lit_eval(train, "stemmed_tokens")
    test_eval = lit_eval(test, "stemmed_tokens")
    val_eval = lit_eval(val, "stemmed_tokens")

    # Remove nonenesense sentences
    train_eval["stemmed_tokens"] = train_eval["stemmed_tokens"].apply(
        remove_no_sentence
    )
    test_eval["stemmed_tokens"] = test_eval["stemmed_tokens"].apply(remove_no_sentence)
    val_eval["stemmed_tokens"] = val_eval["stemmed_tokens"].apply(remove_no_sentence)

    # Drop NAs again
    train_eval.dropna(inplace=True)
    test_eval.dropna(inplace=True)
    val_eval.dropna(inplace=True)

    # Save to csv

    train_eval.to_csv(
        "data/interim/cleaned/train_cleaned_step_1.csv.gz",
        compression="gzip",
        index=False,
    )

    test_eval.to_csv(
        "data/interim/cleaned/test_cleaned_step_1.csv.gz",
        compression="gzip",
        index=False,
    )

    val_eval.to_csv(
        "data/interim/cleaned/val_cleaned_step_1.csv.gz",
        compression="gzip",
        index=False,
    )