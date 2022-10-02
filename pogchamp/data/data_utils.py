import pandas as pd

DATA_PATH = "../corn"
label2id = {
    "pure": 0,
    "broken": 1,
    "silkcut": 2,
    "discolored": 3,
}
id2label = {v: k for k, v in label2id.items()}


def apply_path(row):
    return f"{DATA_PATH}/{row.image}"


def map_label_id(row):
    return label2id[row.label]


def preprocess_dataframe(df, is_test=False):
    df["image"] = df.apply(lambda row: apply_path(row), axis=1)
    if not is_test:
        df["label"] = df.apply(lambda row: map_label_id(row), axis=1)

    return df
