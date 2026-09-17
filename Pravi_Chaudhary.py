import os
import json

DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dataset"
)

JSON_PATH = os.path.join(DATA_ROOT, "pavbhaji.json")

with open(JSON_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

print("Dataset loaded successfully!")
print("Number of JSON records:", len(metadata))
"""
DriveBuddyAI Pav-Bhaji Text Classification Challenge
Pravi Chaudhary

Task:
Predict whether an Instagram post image is Pav-Bhaji (1) or not (0)
using Instagram text/metadata, not image pixels/CNNs.
"""

import os, json, zipfile, re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ZIP_PATH = "dataset.zip"   # point this to the challenge dataset.zip
EXTRACT_DIR = "extracted_dataset"

# ---------- 1. Load / extract ----------
if not os.path.exists(os.path.join(EXTRACT_DIR, "dataset", "pavbhaji.json")):
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(EXTRACT_DIR)

DATA_ROOT = os.path.join(EXTRACT_DIR, "dataset")
JSON_PATH = os.path.join(DATA_ROOT, "pavbhaji.json")

with open(JSON_PATH, encoding="utf-8") as f:
    metadata = json.load(f)

# ---------- 2. Parse JSON metadata ----------
rows = []
for item in metadata:
    url = item.get("display_url", "")
    filename = os.path.basename(url.split("?")[0])

    caption = ""
    try:
        caption = item["edge_media_to_caption"]["edges"][0]["node"]["text"] or ""
    except (KeyError, IndexError, TypeError):
        pass

    rows.append({
        "filename": filename,
        "caption": caption,
        "tags": " ".join(item.get("tags") or []),
        "likes": (item.get("edge_liked_by") or {}).get("count", 0),
        "comments": (item.get("edge_media_to_comment") or {}).get("count", 0),
        "is_video": int(bool(item.get("is_video", False))),
        "timestamp": item.get("taken_at_timestamp", 0),
        "location": str(item.get("location") or "")
    })

meta = pd.DataFrame(rows)

# ---------- 3. Read image-folder labels ----------
image_rows = []
for label in [0, 1]:
    folder = os.path.join(DATA_ROOT, "images", str(label))
    for fn in os.listdir(folder):
        if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_rows.append({"filename": fn, "label": label})

images = pd.DataFrame(image_rows)

# The filename in display_url matches the provided image filename.
df = images.merge(meta, on="filename", how="left")
df["caption"] = df["caption"].fillna("")
df["tags"] = df["tags"].fillna("")

# ---------- 4. Text feature engineering ----------
df["text"] = (
    df["caption"] + " " + df["tags"]
).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

# ---------- 5. Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

# ---------- 6. TF-IDF word + character features ----------
features = FeatureUnion([
    ("word", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        max_features=40000,
        strip_accents="unicode"
    )),
    ("char", TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=40000
    ))
])

model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        C=2
    ))
])

# ---------- 7. Train ----------
model.fit(X_train, y_train)

# ---------- 8. Evaluate ----------
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification report:\n")
print(classification_report(y_test, pred, digits=4))
print("\nConfusion matrix:\n", confusion_matrix(y_test, pred))

# ---------- 9. Predict new text ----------
def predict_post(caption="", tags=""):
    text = (caption + " " + tags).lower()
    label = int(model.predict([text])[0])
    probability = float(model.predict_proba([text])[0, 1])
    return {
        "prediction": "Pav-Bhaji" if label == 1 else "Not Pav-Bhaji",
        "probability": probability
    }

# Example:
# print(predict_post(
#     caption="Butter pav bhaji with extra masala!",
#     tags="pavbhaji streetfood mumbai"
# ))
