import pandas as pd
import matplotlib as plt
import tensorflow_datasets as tfds
import re

"""
#With tfds nightly 
def tfds_converter(name, split):
    ds = tfds.load(name, split=split)
    df = tfds.as_dataframe(ds)
    return df.to_csv("data/external/" + split + ".csv")


if __name__ == "__main__":
    tfds_converter("cnn_dailymail", split="train")
    tfds_converter("cnn_dailymail", split="test")
    tfds_converter("cnn_dailymail", split="validation")
"""
# Load training and validation 'cnn_dailymail' splits

ds_train = tfds.load(name="cnn_dailymail", split="train")
ds_val = tfds.load(name="cnn_dailymail", split="validation")
ds_test = tfds.load(name="cnn_dailymail", split="test")

# Make a function that creates a dataframe
def dataframe_maker(tfdataset, df_name):
    """
    Takes a tensorflow dataset and makes a pandas dataframe
    """
    articles = []
    highlights = []
    # Extract columns from dataset
    for example in tfds.as_numpy(tfdataset):
        article, label = example["article"], example["highlights"]
        articles.append(article)
        highlights.append(label)
    # Assemble the dataframe
    df_name = df_name
    df_name = pd.DataFrame(articles, columns=["article"])
    df_name["highlights"] = highlights
    return df_name


# Convert bytes to string
def string_converter(dataframe):
    """
    Takes a dataframe column and converts it into a proper string
    """
    dataframe["article"] = dataframe["article"].str.decode("utf-8")
    dataframe["highlights"] = dataframe["highlights"].str.decode("utf-8")
    return dataframe


# Cleanup the datasets
# Compose concatenated regex expression to clean data faster
start_of_string = "^\s*"
remove_cnn = ".*\((CNN|EW.com)\)?(\s+--\s+)?"
remove_by_1 = (
    "By \.([^.*]+\.)?([^.]*\. PUBLISHED: \.[^|]*\| \.)? UPDATED:[^.]+\.[^.]+\.\s*"
)
remove_by_2 = "By \.([^.*]+\.)?(\sand[^.*]+\.\s*)?(UPDATED[^.*]+\.[^.*]+\.\s*)?(\slast[^.*]+\.\s*)?"
remove_last_updated = "Last[^.*]+\.\s"
remove_twitter_link = "By \.([^.*]+\.)\s*Follow\s@@[^.*]+\.\s+"
remove_published = "(PUBLISHED[^.*]+\.[^.*]+\.[^.*]+\.\s*)(UPDATED[^.*]+\.[^.*]+\.\s*)?"
# end_of_string = '[\'"]*\s*$'

r_cleanup_source = (
    start_of_string
    + "("
    + "|".join(
        [
            remove_cnn,
            remove_by_1,
            remove_by_2,
            remove_last_updated,
            remove_twitter_link,
            remove_published,
        ]
    )
    + ")"
)

r_cleanup = re.compile(r_cleanup_source)


def cleanup(text):
    # todo replace using r_cleanup
    return r_cleanup.sub("", text)


# Clean and select highlights columns

# Removes newline characters
r_remove_newline_charcter = re.compile("\\n")


def remove_newline(text):
    return r_remove_newline_charcter.sub("", text)


if __name__ == "__main__":
    # Create datasets
    cnn_dm_train = dataframe_maker(ds_train, "cnn_dm_train")
    cnn_dm_val = dataframe_maker(ds_val, "cnn_dm_val")
    cnn_dm_test = dataframe_maker(ds_test, "cnn_dm_test")

    # Convert data to string format
    cnn_dm_train = string_converter(cnn_dm_train)
    cnn_dm_val = string_converter(cnn_dm_val)
    cnn_dm_test = string_converter(cnn_dm_test)

    # Apply the cleanup function to the article column
    cnn_dm_train.article = cnn_dm_train.article.apply(cleanup)
    cnn_dm_val.article = cnn_dm_val.article.apply(cleanup)
    cnn_dm_test.article = cnn_dm_test.article.apply(cleanup)

    # Clean highlights of training set
    cnn_dm_train.highlights = cnn_dm_train.highlights.apply(remove_newline)

    # Clean highlights of validation set
    cnn_dm_val.highlights = cnn_dm_val.highlights.apply(remove_newline)

    # Clean highlights of testing set
    cnn_dm_test.highlights = cnn_dm_test.highlights.apply(remove_newline)

    # Save cleaned data to compressed format in interim folder
    cnn_dm_train.to_csv("data/interim/cnn_dm_train.csv.gz", compression="gzip")
    cnn_dm_val.to_csv("data/interim/cnn_dm_val.csv.gz", compression="gzip")
    cnn_dm_test.to_csv("data/interim/cnn_dm_test.csv.gz", compression="gzip")
