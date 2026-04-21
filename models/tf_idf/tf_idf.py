import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

TRAIN_PATH = "datasets/no_handcrafted/mental_health_text_train.csv"
TEST_PATH  = "datasets/no_handcrafted/mental_health_text_test.csv"

df_train = pd.read_csv(TRAIN_PATH, index_col=0)
df_test  = pd.read_csv(TEST_PATH,  index_col=0)

print(f"Train size: {len(df_train)}, Test size: {len(df_test)}")
print(f"Classes: {sorted(df_train['status'].unique())}\n")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words='english'
)

X_train = vectorizer.fit_transform(df_train['text'])
X_test  = vectorizer.transform(df_test['text'])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_train['status'])
y_test  = label_encoder.transform(df_test['status'])

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced',
    n_jobs=1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
print(cm_df)


def predict_status(texts: list[str]) -> list[str]:
    """Return predicted mental health status for a list of text strings."""
    X = vectorizer.transform(texts)
    preds = model.predict(X)
    return label_encoder.inverse_transform(preds).tolist()


if __name__ == "__main__":
    examples = [
        "I can't stop crying and I don't know why. Everything feels hopeless.",
        "Had a great day today, feeling really good about life!",
        "My heart keeps racing and I can't calm down no matter what I try.",
        "I don't see the point anymore. Nobody would miss me.",
    ]
    print("\nSample predictions:")
    for text, label in zip(examples, predict_status(examples)):
        print(f"  [{label:>11}]  {text[:70]}")