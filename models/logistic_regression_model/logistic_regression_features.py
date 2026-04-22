import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.sparse import hstack
from nltk.tokenize import word_tokenize
import string


TRAIN_PATH = "../../datasets/handcrafted/mental_health_text_train_features.csv"
TEST_PATH  = "../../datasets/handcrafted/mental_health_text_test_features.csv"

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


def train_log_regression(vectorizer):
    df_train = pd.read_csv(TRAIN_PATH, index_col=0)
    df_test  = pd.read_csv(TEST_PATH,  index_col=0)

    print(f"Train size: {len(df_train):,}, Test size: {len(df_test):,}")
    print(f"Classes: {sorted(df_train['status'].unique())}\n")

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
    y_test = label_encoder.transform(df_test['status'])

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    
    # create graphs for precision, recall, f1 score per class
    print("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
    classes = label_encoder.classes_

    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]

    x = np.arange(len(classes))
    width = 0.25
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    plt.figure()
    plt.bar(x - width, precision, width, label='Precision', color=colors[0])
    plt.bar(x, recall, width, label='Recall', color=colors[1])
    plt.bar(x + width, f1, width, label='F1 Score', color=colors[2])
    plt.xticks(x, classes, rotation=45)
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.title(f"Per-Class Evaluation Scores (Accuracy: {accuracy:.4f} )")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # print confusion matrix
    print("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print(cm_df)

    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_

    # print top 5 words most important words for each class
    for i, class_label in enumerate(label_encoder.classes_):
        top_indices = coefs[i].argsort()[-5:]
        
        plt.figure()
        plt.barh(feature_names[top_indices], coefs[i][top_indices])
        plt.title(f"Top Words for Class: {class_label}")
        plt.xlabel("Importance")
        plt.ylabel("Words")
        plt.tight_layout()
        plt.show()
    
    return model, label_encoder

def predict_status(texts: list[str], model: LogisticRegression, vectorizer: CountVectorizer, label_encoder: LabelEncoder) -> list[str]:
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
    # this implementation is a basic logistic regression which uses a BoW vectorizer
    vectorizer = CountVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    model, label_encoder = train_log_regression(vectorizer)

    inference_examples = [
        "I can't stop crying and I don't know why. Everything feels hopeless.",
        "Had a great day today, feeling really good about life!",
        "My heart keeps racing and I can't calm down no matter what I try.",
        "I don't see the point anymore. Nobody would miss me.",
    ]
    print("Test predictions:"),
    for text, label in zip(inference_examples, predict_status(inference_examples, model, vectorizer, label_encoder)):
        print(f"  [{label:>11}]  {text[:70]}")
