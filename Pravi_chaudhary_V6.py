# ============================================================
# DRIVEBUDDYAI PAV-BHAJI CLASSIFICATION - VERSION 6
# DATA-DRIVEN MODEL SELECTION
# Candidate: Pravi Chaudhary
#
# IMPORTANT:
# Text/metadata classification only.
# No image pixels.
# No filename.
# No label folder as a feature.
# No owner/account ID.
# ============================================================

import os
import json
import zipfile
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import FeatureUnion

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. FIND DATASET
# ============================================================

HERE = os.path.dirname(
    os.path.abspath(__file__)
)


def find_dataset(root):

    for current, dirs, files in os.walk(root):

        if (
            "pavbhaji.json" in files
            and "images" in dirs
        ):

            return current

    return None


DATASET_DIR = find_dataset(HERE)


# If dataset is not extracted
if DATASET_DIR is None:

    ZIP_PATH = os.path.join(
        HERE,
        "dataset.zip"
    )

    if os.path.exists(ZIP_PATH):

        EXTRACT_DIR = os.path.join(
            HERE,
            "_dataset_v6"
        )

        os.makedirs(
            EXTRACT_DIR,
            exist_ok=True
        )

        with zipfile.ZipFile(
            ZIP_PATH,
            "r"
        ) as z:

            z.extractall(
                EXTRACT_DIR
            )

        DATASET_DIR = find_dataset(
            EXTRACT_DIR
        )


if DATASET_DIR is None:

    raise FileNotFoundError(
        """
Dataset not found.

Required structure:

dataset/
    pavbhaji.json
    images/
        0/
        1/
"""
    )


print("=" * 70)
print("DRIVEBUDDYAI PAV-BHAJI CLASSIFIER V6")
print("=" * 70)

print(
    "Dataset:",
    DATASET_DIR
)


# ============================================================
# 2. LOAD JSON
# ============================================================

JSON_PATH = os.path.join(
    DATASET_DIR,
    "pavbhaji.json"
)


with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:

    raw = json.load(f)


print(
    "JSON records:",
    len(raw)
)


# ============================================================
# 3. CAPTION EXTRACTION
# ============================================================

def get_caption(item):

    try:

        return (
            item[
                "edge_media_to_caption"
            ][
                "edges"
            ][0][
                "node"
            ][
                "text"
            ] or ""
        )

    except:

        return ""


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text).lower()

    # URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Preserve word information
    text = text.replace(
        "#",
        " hashtag_"
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-zA-Z0-9_\s]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 5. EXTRACT METADATA
# ============================================================

metadata = []


for item in raw:

    display_url = str(
        item.get(
            "display_url",
            ""
        )
    )

    filename = os.path.basename(
        display_url.split("?")[0]
    )

    caption = get_caption(
        item
    )

    tags = item.get(
        "tags",
        []
    )

    if tags is None:

        tags = []


    tags_text = " ".join(
        str(x)
        for x in tags
    )


    metadata.append({

        "filename":
            filename,

        "caption":
            caption,

        "tags":
            tags_text

    })


meta = pd.DataFrame(
    metadata
)


# ============================================================
# 6. LOAD IMAGE LABELS
# ============================================================

image_rows = []


for label in [0, 1]:

    folder = os.path.join(

        DATASET_DIR,

        "images",

        str(label)

    )


    if not os.path.exists(
        folder
    ):

        raise FileNotFoundError(
            folder
        )


    for filename in os.listdir(
        folder
    ):

        if filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            )
        ):

            image_rows.append({

                "filename":
                    filename,

                "label":
                    label

            })


images = pd.DataFrame(
    image_rows
)


# ============================================================
# 7. JOIN DATA
# ============================================================

df = images.merge(

    meta,

    on="filename",

    how="left"

)


df["caption"] = (
    df["caption"]
    .fillna("")
    .astype(str)
)


df["tags"] = (
    df["tags"]
    .fillna("")
    .astype(str)
)


# ============================================================
# 8. CLEAN TEXT
# ============================================================

df["caption_clean"] = (
    df["caption"]
    .apply(clean_text)
)


df["tags_clean"] = (
    df["tags"]
    .apply(clean_text)
)


# ============================================================
# 9. DIFFERENT TEXT REPRESENTATIONS
# ============================================================

# Representation 1
df["CAPTION_ONLY"] = (
    df["caption_clean"]
)


# Representation 2
df["TAGS_ONLY"] = (
    df["tags_clean"]
)


# Representation 3
df["CAPTION_TAGS"] = (

    df["caption_clean"]
    + " "
    + df["tags_clean"]

)


# Representation 4
# Repeat caption and tags to give
# important fields more weight.

df["WEIGHTED_TEXT"] = (

    df["caption_clean"]
    + " "
    + df["caption_clean"]
    + " "

    + df["tags_clean"]
    + " "
    + df["tags_clean"]

)


# ============================================================
# 10. DATA SUMMARY
# ============================================================

print("\nDataset Summary")

print(
    "Total:",
    len(df)
)

print(
    "\nLabels:"
)

print(
    df["label"]
    .value_counts()
    .sort_index()
)


print(
    "\nEmpty captions:",
    (
        df["caption_clean"]
        .str.len()
        == 0
    ).sum()
)


print(
    "Empty tags:",
    (
        df["tags_clean"]
        .str.len()
        == 0
    ).sum()
)


# ============================================================
# 11. TRAIN TEST SPLIT
# ============================================================

indices = np.arange(
    len(df)
)


train_idx, test_idx = train_test_split(

    indices,

    test_size=0.20,

    stratify=df["label"],

    random_state=42

)


y_train = (
    df.loc[
        train_idx,
        "label"
    ]
)


y_test = (
    df.loc[
        test_idx,
        "label"
    ]
)


# ============================================================
# 12. TF-IDF FUNCTION
# ============================================================

def create_features(

    train_text,

    test_text,

    word_ngram,

    char_ngram,

    word_max,

    char_max

):

    word = TfidfVectorizer(

        analyzer="word",

        ngram_range=word_ngram,

        min_df=1,

        max_df=0.98,

        sublinear_tf=True,

        max_features=word_max,

        strip_accents="unicode"

    )


    char = TfidfVectorizer(

        analyzer="char_wb",

        ngram_range=char_ngram,

        min_df=1,

        max_df=0.995,

        sublinear_tf=True,

        max_features=char_max

    )


    X_train_word = (
        word.fit_transform(
            train_text
        )
    )


    X_test_word = (
        word.transform(
            test_text
        )
    )


    X_train_char = (
        char.fit_transform(
            train_text
        )
    )


    X_test_char = (
        char.transform(
            test_text
        )
    )


    X_train = (
        np.hstack(
            [
                X_train_word.toarray(),
                X_train_char.toarray()
            ]
        )
    )


    X_test = (
        np.hstack(
            [
                X_test_word.toarray(),
                X_test_char.toarray()
            ]
        )
    )


    return (
        X_train,
        X_test
    )


# ============================================================
# 13. MODEL SEARCH
# ============================================================

representations = [

    "CAPTION_ONLY",

    "TAGS_ONLY",

    "CAPTION_TAGS",

    "WEIGHTED_TEXT"

]


models = []


# Logistic Regression

for C in [

    0.01,
    0.03,
    0.1,
    0.3,
    0.5,
    1,
    2,
    5,
    10

]:

    models.append({

        "name":
            f"Logistic C={C}",

        "model":
            LogisticRegression(

                C=C,

                max_iter=5000,

                class_weight=None,

                solver="liblinear"

            )

    })


# Logistic with balanced

for C in [

    0.1,
    0.3,
    1,
    3,
    10

]:

    models.append({

        "name":
            f"Balanced Logistic C={C}",

        "model":
            LogisticRegression(

                C=C,

                max_iter=5000,

                class_weight="balanced",

                solver="liblinear"

            )

    })


# SVM

for C in [

    0.01,
    0.03,
    0.1,
    0.3,
    0.5,
    1,
    2,
    5,
    10

]:

    models.append({

        "name":
            f"SVM C={C}",

        "model":
            LinearSVC(

                C=C,

                class_weight=None,

                max_iter=20000

            )

    })


# ============================================================
# 14. CROSS VALIDATION SEARCH
# ============================================================

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


search_results = []


print(
    "\n"
    + "=" * 70
)

print(
    "DATA-DRIVEN 5-FOLD MODEL SEARCH"
)

print(
    "=" * 70
)


for representation in representations:

    print(
        "\nTesting:",
        representation
    )


    text = df[
        representation
    ]


    X_train_text = (
        text.iloc[
            train_idx
        ]
    )


    X_test_text = (
        text.iloc[
            test_idx
        ]
    )


    # --------------------------------------------------------
    # Different feature configurations
    # --------------------------------------------------------

    feature_configs = [

        {

            "word":
                (1, 1),

            "char":
                (3, 5)

        },

        {

            "word":
                (1, 2),

            "char":
                (3, 5)

        },

        {

            "word":
                (1, 3),

            "char":
                (3, 6)

        },

        {

            "word":
                (1, 4),

            "char":
                (3, 7)

        }

    ]


    for config in feature_configs:

        print(
            " Features:",
            config
        )


        X_train, X_test = (
            create_features(

                X_train_text,

                X_test_text,

                config["word"],

                config["char"],

                30000,

                30000

            )
        )


        # ----------------------------------------------------
        # Evaluate models
        # ----------------------------------------------------

        for entry in models:

            model = (
                entry["model"]
            )


            scores = (
                cross_val_score(

                    model,

                    X_train,

                    y_train,

                    cv=cv,

                    scoring="accuracy"

                )
            )


            cv_accuracy = (
                scores.mean()
            )


            # Train on training split
            model.fit(

                X_train,

                y_train

            )


            prediction = (
                model.predict(
                    X_test
                )
            )


            holdout_accuracy = (
                accuracy_score(

                    y_test,

                    prediction

                )
            )


            holdout_f1 = (
                f1_score(

                    y_test,

                    prediction,

                    zero_division=0

                )
            )


            search_results.append({

                "representation":
                    representation,

                "word_ngram":
                    str(config["word"]),

                "char_ngram":
                    str(config["char"]),

                "model":
                    entry["name"],

                "cv_accuracy":
                    cv_accuracy,

                "holdout_accuracy":
                    holdout_accuracy,

                "holdout_f1":
                    holdout_f1

            })


# ============================================================
# 15. RESULTS
# ============================================================

results_df = pd.DataFrame(
    search_results
)


results_df = (
    results_df
    .sort_values(
        [
            "cv_accuracy",
            "holdout_accuracy"
        ],
        ascending=False
    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "TOP 20 CONFIGURATIONS"
)

print(
    "=" * 70
)


print(

    results_df
    .head(20)
    .to_string(
        index=False
    )

)


# ============================================================
# 16. SELECT BEST MODEL
# ============================================================

best = (
    results_df
    .iloc[0]
)


print(
    "\n"
    + "=" * 70
)

print(
    "BEST DATA-DRIVEN CONFIGURATION"
)

print(
    "=" * 70
)


print(
    best.to_string()
)


# ============================================================
# 17. TRAIN BEST MODEL AGAIN
# ============================================================

best_rep = (
    best["representation"]
)


best_word = eval(
    best["word_ngram"]
)


best_char = eval(
    best["char_ngram"]
)


# Extract C
best_model_name = (
    best["model"]
)


if "Balanced Logistic" in best_model_name:

    best_C = float(
        best_model_name
        .split("=")[1]
    )

    final_model = (
        LogisticRegression(

            C=best_C,

            max_iter=10000,

            class_weight="balanced",

            solver="liblinear"

        )
    )


elif "Logistic" in best_model_name:

    best_C = float(
        best_model_name
        .split("=")[1]
    )

    final_model = (
        LogisticRegression(

            C=best_C,

            max_iter=10000,

            class_weight=None,

            solver="liblinear"

        )
    )


else:

    best_C = float(
        best_model_name
        .split("=")[1]
    )

    final_model = (
        LinearSVC(

            C=best_C,

            class_weight=None,

            max_iter=20000

        )
    )


best_text = (
    df[best_rep]
)


X_train_text = (
    best_text.iloc[
        train_idx
    ]
)


X_test_text = (
    best_text.iloc[
        test_idx
    ]
)


X_train, X_test = (
    create_features(

        X_train_text,

        X_test_text,

        best_word,

        best_char,

        50000,

        50000

    )
)


final_model.fit(

    X_train,

    y_train

)


final_prediction = (
    final_model.predict(
        X_test
    )
)


# ============================================================
# 18. FINAL EVALUATION
# ============================================================

accuracy = (
    accuracy_score(

        y_test,

        final_prediction

    )
)


precision = (
    precision_score(

        y_test,

        final_prediction,

        zero_division=0

    )
)


recall = (
    recall_score(

        y_test,

        final_prediction,

        zero_division=0

    )
)


f1 = (
    f1_score(

        y_test,

        final_prediction,

        zero_division=0

    )
)


print(
    "\n"
    + "=" * 70
)

print(
    "FINAL HOLDOUT RESULT"
)

print(
    "=" * 70
)


print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)


print(
    "Precision:",
    round(
        precision,
        4
    )
)


print(
    "Recall:",
    round(
        recall,
        4
    )
)


print(
    "F1:",
    round(
        f1,
        4
    )
)


print(
    "\nClassification Report:"
)


print(
    classification_report(

        y_test,

        final_prediction,

        digits=4

    )
)


print(
    "\nConfusion Matrix:"
)


print(
    confusion_matrix(

        y_test,

        final_prediction

    )
)


# ============================================================
# 19. ERROR ANALYSIS
# ============================================================

error_analysis = df.iloc[
    test_idx
].copy()


error_analysis[
    "actual"
] = y_test.values


error_analysis[
    "predicted"
] = final_prediction


error_analysis[
    "correct"
] = (

    error_analysis[
        "actual"
    ]

    ==

    error_analysis[
        "predicted"
    ]

)


false_positive = (
    error_analysis[
        (
            error_analysis["actual"]
            == 0
        )
        &
        (
            error_analysis["predicted"]
            == 1
        )
    ]
)


false_negative = (
    error_analysis[
        (
            error_analysis["actual"]
            == 1
        )
        &
        (
            error_analysis["predicted"]
            == 0
        )
    ]
)


print(
    "\nFalse Positives:",
    len(false_positive)
)


print(
    "False Negatives:",
    len(false_negative)
)


# ============================================================
# 20. SAVE RESULTS
# ============================================================

results_df.to_csv(

    os.path.join(
        HERE,
        "V6_all_model_results.csv"
    ),

    index=False

)


error_analysis[
    [
        "filename",
        "caption",
        "tags",
        "actual",
        "predicted",
        "correct"
    ]
].to_csv(

    os.path.join(
        HERE,
        "V6_error_analysis.csv"
    ),

    index=False

)


print(
    "\nFiles created:"
)

print(
    "V6_all_model_results.csv"
)

print(
    "V6_error_analysis.csv"
)


print(
    "\n"
    + "=" * 70
)

print(
    "V6 COMPLETE"
)

print(
    "=" * 70
)