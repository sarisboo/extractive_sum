import pandas as pd
import numpy as np
from ast import literal_eval
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import time

nltk.download("averaged_perceptron_tagger")

# print(nltk.help.upenn_tagset("VBG"))

# Extract Singular and plural proper nouns
def proper_noun_count_extractor(word_list):
    """
    Accepts a list of words as input and returns
    counts of NNP and NNPS words present
    """
    tagged_sentence = pos_tag(literal_eval(word_list))

    # Count Proper Nouns
    proper_noun_count = len(
        [word for word, pos in tagged_sentence if pos in ["NNP", "NNPS"]]
    )
    return proper_noun_count


def verb_count_extractor(sentence):
    """Accepts a string as input and tokenizes and
    counts the number of verbs in the string"""
    tagged_sentence = pos_tag(word_tokenize(sentence))

    # Count Verbs
    verb_count = len(
        [
            word
            for word, pos in tagged_sentence
            if pos in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        ]
    )
    return verb_count


def pronoun_count(sentence):
    """Accepts string as input and counts teh number of pronouns present"""

    tagged_sentence = pos_tag(word_tokenize(sentence))

    # Count Pronouns
    pron_count = len([word for word, pos in tagged_sentence if pos in ["PRON"]])
    return pron_count


def sentence_length_extractor(sentence):
    """
    Accepts string as input, tokenizes it excluding puntuation and counts the words
    """
    # splits without punctuatiom
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    only_words_count = len(tokenizer.tokenize(sentence))

    return only_words_count


if __name__ == "__main__":

    start_time = time.time()
    # Load Data (crpped the dataset for faster processing)
    train = pd.read_csv(
        "data/interim/cleaned/train_cleaned_step_1.csv.gz",
        compression="gzip",
        nrows=500000,
    )
    test = pd.read_csv(
        "data/interim/cleaned/test_cleaned_step_1.csv.gz", compression="gzip"
    )
    val = pd.read_csv(
        "data/interim/cleaned/val_cleaned_step_1.csv.gz", compression="gzip"
    )

    # Create proper noun count column
    train["proper_noun_count"] = train["no_stop_words_tokens"].apply(
        lambda x: proper_noun_count_extractor(x)
    )

    # Create verb count column
    train["verb_count"] = train["sentence"].apply(lambda x: verb_count_extractor(x))

    # create pronoun count column
    train["pron_count"] = train["sentence"].apply(lambda x: pronoun_count(x))

    # Create verb count column
    train["sentence_len"] = train["sentence"].apply(
        lambda x: sentence_length_extractor(x)
    )

    train.to_csv(
        "src/features/cropped/propernouns_verbs_sent_len/train_pn_verb_counts.csv.gz",
        compression="gzip",
        index=False,
    )

    test.to_csv(
        "src/features/cropped/propernouns_verbs_sent_len/test_pn_verb_counts.csv.gz",
        compression="gzip",
        index=False,
    )

    val.to_csv(
        "src/features/cropped/propernouns_verbs_sent_len/val_pn_verb_counts.csv.gz",
        compression="gzip",
        index=False,
    )
    print(time.time() - start_time)