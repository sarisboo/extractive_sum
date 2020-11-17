# Import necessary Modules
import pandas as pd
import numpy as np
from ast import literal_eval

# Create a functions that label sentences with question marks, exlamation points and quotes
def question_mark_finder(sentence):
    """
    Returns 1 if sentence contains question mark, 0 otherwise
    """
    if "?" in sentence:
        return 1
    else:
        return 0


def exclamation_mark_finder(sentence):
    """
    Returns 1 if sentence contains question mark, 0 otherwise
    """
    if "!" in sentence:
        return 1
    else:
        return 0


def quotes_finder(sentence):
    """
    Returns 1 if sentence contains question mark, 0 otherwise
    """
    if "'" or "`" in sentence:
        return 1
    else:
        return 0


# Apply these functions to the dataset
if __name__ == "__main__":

    # Load Data
    train = pd.read_csv(
        "../data/interim/cleaned/train_cleaned_step_1.csv.gz", compression="gzip"
    )
    test = pd.read_csv(
        "../data/interim/cleaned/test_cleaned_step_1.csv.gz", compression="gzip"
    )
    val = pd.read_csv(
        "../data/interim/cleaned/val_cleaned_step_1.csv.gz", compression="gzip"
    )

    # Create question mark column
    train["question_mark"] = train["pass"].apply(lambda x: question_mark_finder(x))
    test["question_mark"] = test["pass"].apply(lambda x: question_mark_finder(x))
    val["question_mark"] = val["pass"].apply(lambda x: question_mark_finder(x))

    # Create exclamation mark column
    train["exclamation_mark"] = train["pass"].apply(
        lambda x: exclamation_mark_finder(x)
    )
    test["exclamation_mark"] = test["pass"].apply(lambda x: exclamation_mark_finder(x))
    val["exclamation_mark"] = val["pass"].apply(lambda x: exclamation_mark_finder(x))

    # Create quote finder column
    train["quote_marks"] = train["pass"].apply(lambda x: exclamation_mark_finder(x))
    test["quote_marks"] = test["pass"].apply(lambda x: exclamation_mark_finder(x))
    val["quote_marks"] = val["pass"].apply(lambda x: exclamation_mark_finder(x))

    train.to_csv(
        "src/features/propernouns_verbs_sent_len/train_pn_verb_counts_punctuation.csv.gz",
        compression="gzip",
        index=False,
    )

    test.to_csv(
        "src/features/propernouns_verbs_sent_len/test_pn_verb_counts_punctuation.csv.gz",
        compression="gzip",
        index=False,
    )

    val.to_csv(
        "src/features/propernouns_verbs_sent_len/val_pn_verb_counts_punctuation.csv.gz",
        compression="gzip",
        index=False,
    )