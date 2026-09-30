# ============================================================
# AI-Powered SMS Spam & Scam Detector
# NLP + Machine Learning
# ============================================================

import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("AI-POWERED SMS SPAM & SCAM DETECTOR")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1"
)
print("Dataset loaded successfully!")
print("Original shape:", df.shape)

print("\nOriginal columns:")
print(df.columns.tolist())


# ============================================================
# 2. IDENTIFY IMPORTANT COLUMNS
# ============================================================

# Different Kaggle datasets may use different column names.

message_columns = [
    "message",
    "Message",
    "text",
    "Text",
    "sms",
    "SMS",
    "v2"
]

label_columns = [
    "label",
    "Label",
    "category",
    "Category",
    "type",
    "v1"
]

message_column = None
label_column = None


# Find message column
for col in message_columns:
    if col in df.columns:
        message_column = col
        break


# Find label column
for col in label_columns:
    if col in df.columns:
        label_column = col
        break


# If automatic detection fails
if message_column is None or label_column is None:

    print("\nAutomatic column detection failed.")

    # Common SMS Spam Collection format:
    # first column = label
    # second column = message

    label_column = df.columns[0]
    message_column = df.columns[1]


print("\nSelected label column:", label_column)
print("Selected message column:", message_column)


# ============================================================
# 3. CREATE CLEAN DATAFRAME
# ============================================================

df = df[[label_column, message_column]].copy()

df.columns = ["label", "message"]

print("\nData preview:")
print(df.head())


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

df.dropna(inplace=True)

print("\nRows after removing missing values:", len(df))


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

df.drop_duplicates(subset="message", inplace=True)

print("Rows after removing duplicates:", len(df))


# ============================================================
# 6. CLEAN LABELS
# ============================================================

df["label"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)

print("\nUnique labels before conversion:")
print(df["label"].unique())


# Convert labels:
# ham / legitimate / legit = 0
# spam / scam = 1

label_mapping = {
    "ham": 0,
    "legitimate": 0,
    "legit": 0,
    "normal": 0,

    "spam": 1,
    "scam": 1
}

df["label"] = df["label"].map(label_mapping)


# Remove unknown labels
df.dropna(subset=["label"], inplace=True)

df["label"] = df["label"].astype(int)


print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# 7. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):
    """
    Clean SMS text before machine learning.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCleaning SMS messages...")

df["clean_message"] = df["message"].apply(clean_text)

print("Text cleaning completed.")


# ============================================================
# 8. FEATURES AND TARGET
# ============================================================

X = df["clean_message"]
y = df["label"]


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 10. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print(
    "TF-IDF features:",
    X_train_tfidf.shape[1]
)


# ============================================================
# 11. MODEL 1 - NAIVE BAYES
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1: MULTINOMIAL NAIVE BAYES")
print("=" * 60)

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(
    X_test_tfidf
)


# ============================================================
# 12. MODEL 2 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2: LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(
    X_test_tfidf
)


# ============================================================
# 13. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    print("\n" + "-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "LEGITIMATE",
                "SPAM"
            ],
            zero_division=0
        )
    )

    return accuracy, precision, recall, f1


# ============================================================
# 14. EVALUATE BOTH MODELS
# ============================================================

nb_results = evaluate_model(
    "Multinomial Naive Bayes",
    y_test,
    nb_predictions
)

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)


# ============================================================
# 15. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Naive Bayes",
        "Logistic Regression"
    ],

    "Accuracy": [
        nb_results[0],
        lr_results[0]
    ],

    "Precision": [
        nb_results[1],
        lr_results[1]
    ],

    "Recall": [
        nb_results[2],
        lr_results[2]
    ],

    "F1-Score": [
        nb_results[3],
        lr_results[3]
    ]
})


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    lr_predictions
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "LEGITIMATE",
        "SPAM"
    ],
    yticklabels=[
        "LEGITIMATE",
        "SPAM"
    ]
)

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. MODEL COMPARISON GRAPH
# ============================================================

comparison_plot = comparison.set_index(
    "Model"
)

comparison_plot.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title(
    "Machine Learning Model Comparison"
)

plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "model_comparison.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. REAL-TIME SMS PREDICTION
# ============================================================

def predict_sms(message):

    # Clean message
    cleaned_message = clean_text(message)

    # Convert text into TF-IDF
    message_vector = vectorizer.transform(
        [cleaned_message]
    )

    # Prediction
    prediction = lr_model.predict(
        message_vector
    )[0]

    # Probability
    probabilities = lr_model.predict_proba(
        message_vector
    )[0]

    confidence = max(probabilities) * 100

    if prediction == 1:
        result = "SPAM / SCAM"
    else:
        result = "LEGITIMATE"

    print("\n" + "=" * 60)
    print("SMS PREDICTION")
    print("=" * 60)

    print("Message:")
    print(message)

    print("\nPrediction:", result)

    print(
        f"Confidence: {confidence:.2f}%"
    )


# ============================================================
# 19. TEST EXAMPLES
# ============================================================

predict_sms(
    "Congratulations! You have won a £1000 prize. Click now!"
)

predict_sms(
    "Hey, are we still meeting at 6 pm?"
)


# ============================================================
# 20. USER INPUT
# ============================================================

print("\n" + "=" * 60)
print("TRY YOUR OWN SMS")
print("=" * 60)

user_message = input(
    "\nEnter an SMS message: "
)

predict_sms(user_message)

print("\nProject completed successfully!")