import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

TRAIN_PATH = "../../datasets/no_handcrafted/mental_health_text_train.csv"
TEST_PATH  = "../../datasets/no_handcrafted/mental_health_text_test.csv"

def train_log_regression(vectorizer):
    df_train = pd.read_csv(TRAIN_PATH, index_col=0)
    df_test  = pd.read_csv(TEST_PATH,  index_col=0)

    print(f"Train size: {len(df_train):,}, Test size: {len(df_test):,}")
    print(f"Classes: {sorted(df_train['status'].unique())}\n")

    X_train = vectorizer.fit_transform(df_train['text'])
    X_test = vectorizer.transform(df_test['text'])

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
    X = vectorizer.transform(texts)
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
