import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle


# Read the data
# Training set rediced size
train = pd.read_csv(
    "data/interim/cleaned/train_cleaned_step_1.csv.gz",
    compression="gzip",
    nrows=1800000,
)
test = pd.read_csv(
    "data/interim/cleaned/test_cleaned_step_1.csv.gz", compression="gzip"
)
val = pd.read_csv("data/interim/cleaned/val_cleaned_step_1.csv.gz", compression="gzip")

# Drop Nas
train.dropna(inplace=True)
test.dropna(inplace=True)
val.dropna(inplace=True)

# Use training set to compute tfidf as part of feature engineering
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 1), analyzer="word")
corpus_train = train["sentence"]
corpus_test = test["sentence"]
corpus_val = val["sentence"]

# Fit transform train and transform test and val set using the same vectorizer
train_sparse_matrix = vectorizer.fit_transform(corpus_train)
test_sparse_matrix = vectorizer.transform(corpus_test)
val_sparse_matrix = vectorizer.transform(corpus_val)

if __name__ == "__main__":
    # save cropped dataset
    train.to_csv(
        "src/features/cropped/train_cropped.csv.gz",
        compression="gzip",
        index=False,
    )
    # Save tfidf sparse matrix for each dataset
    sparse.save_npz(
        "src/features/cropped/tf_idf_feature/train_sparse_matrix.npz",
        train_sparse_matrix,
    )

    sparse.save_npz(
        "src/features/cropped/tf_idf_feature/test_sparse_matrix.npz", test_sparse_matrix
    )

    sparse.save_npz(
        "src/features/cropped/tf_idf_feature/val_sparse_matrix.npz", val_sparse_matrix
    )

    # Pickle vectorizer
    pickle.dump(
        vectorizer,
        open("src/features/cropped/tf_idf_feature/tfidf_vectorizer.pickle", "wb"),
    )
