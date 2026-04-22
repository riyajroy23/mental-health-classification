import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import word_tokenize
import string


TRAIN_PATH = "datasets/handcrafted/mental_health_text_train_features.csv"
TEST_PATH  = "datasets/handcrafted/mental_health_text_test_features.csv"

df_train = pd.read_csv(TRAIN_PATH, index_col=0)
df_test  = pd.read_csv(TEST_PATH,  index_col=0)

print(f"Train size: {len(df_train):,}, Test size: {len(df_test):,}")
print(f"Classes: {sorted(df_train['status'].unique())}\n")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words='english'
)

handcrafted_names = df_train.drop(columns=['text', 'status']).columns

X_train_sparse = vectorizer.fit_transform(df_train['text'])
X_test_sparse = vectorizer.transform(df_test['text'])

scaler = StandardScaler()
X_train_hcf_scaled = scaler.fit_transform(df_train[handcrafted_names])
X_test_hcf_scaled = scaler.transform(df_test[handcrafted_names])

X_train = hstack([X_train_sparse, X_train_hcf_scaled])
X_test = hstack([X_test_sparse, X_test_hcf_scaled])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(df_train['status'])
y_test  = label_encoder.transform(df_test['status'])

model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced',
    n_jobs=-1
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

def count_from_list(tokens, word_list):
  count = 0
  
  for token in tokens:
    if token in word_list:
      count += 1

  return count

def count_punctuation(tokens):
  count = 0
  for token in tokens:
    if token in string.punctuation:
      count += 1
  return count

DEP_WORDS = ['like', 'feel', 'know', 'want', 'get', 'time', 'life', 'even', 'really', 'would', 'people', 'one', 'things', 'much', 'go', 'never', 'think', 'years', 'going', 'day', 'help', 'back', 'could', 'still', 'friends']
HAPPY_WORDS = ['happy', 'got', 'made', 'went', 'time', 'new', 'day', 'work', 'last', 'friend', 'good', 'really', 'one', 'able', 'today', 'friends', 'family', 'first', 'home', 'get', 'found', 'yesterday', 'son', 'great', 'night']

def predict_status(texts: list[str]) -> list[str]:
    """Return predicted mental health status for a list of text strings."""
    X_sparse = vectorizer.transform(texts)
    
    # Handcrafted features
    text_len = []
    token_count = []
    punct_count = []
    dep_count = []
    happy_count = []

    for text in texts:
        tokens = word_tokenize(text)

        text_len.append(len(text))
        token_count.append(len(tokens))
        punct_count.append(count_punctuation(tokens))
        dep_count.append(count_from_list(tokens, DEP_WORDS))
        happy_count.append(count_from_list(tokens, HAPPY_WORDS))

    handcrafted = np.array([text_len, token_count, punct_count, dep_count, happy_count]).T

    # Combine sparse + handcrafted
    X = hstack([X_sparse, handcrafted])

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
