import os
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt

LABELS = ['Anxiety', 'Depression', 'Normal', 'Suicidal']
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 4
LR = 2e-5
MODEL_NAME = "bert-base-uncased"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'no_handcrafted')
SAVE_PATH = os.path.join(os.path.dirname(__file__), 'mental_health_bert.pt')
GRAPHS_DIR = os.path.join(os.path.dirname(__file__), 'graphs')

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'mental_health_text_train.csv'))
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'mental_health_text_test.csv'))

    le = LabelEncoder()
    le.fit(LABELS)

    train_df = train_df.dropna(subset=['text', 'status'])
    test_df = test_df.dropna(subset=['text', 'status'])

    train_labels = le.transform(train_df['status'])
    test_labels = le.transform(test_df['status'])

    return train_df['text'], train_labels, test_df['text'], test_labels, le


def train(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(loader)


def evaluate(model, loader, le):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=le.classes_, output_dict=True)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
    return acc, report


def plot_training_loss(losses):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker='o')
    plt.title("Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(range(1, len(losses) + 1))
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'training_loss.png'))
    plt.close()
    print("Saved bert_model/graphs/training_loss.png")


def plot_eval_scores(report, acc):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(LABELS))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        scores = [report[cls][metric] for cls in LABELS]
        ax.bar(x + i * width, scores, width, label=metric.capitalize())

    ax.set_title(f"Per-Class Evaluation Scores (Accuracy: {acc:.4f})")
    ax.set_xticks(x + width)
    ax.set_xticklabels(LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'eval_scores.png'))
    plt.close()
    print("Saved bert_model/graphs/eval_scores.png")


def main():
    train_texts, train_labels, test_texts, test_labels, le = load_data()

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = MentalHealthDataset(train_texts, train_labels, tokenizer)
    test_dataset = MentalHealthDataset(test_texts, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    losses = []
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train(model, train_loader, optimizer, scheduler)
        losses.append(avg_loss)
        print(f"Epoch {epoch}/{EPOCHS} — Loss: {avg_loss:.4f}")

    print("\n--- Test Set Evaluation ---")
    acc, report = evaluate(model, test_loader, le)

    plot_training_loss(losses)
    plot_eval_scores(report, acc)

    torch.save({
        'model_state_dict': model.state_dict(),
        'label_classes': le.classes_.tolist(),
        'max_len': MAX_LEN,
        'model_name': MODEL_NAME
    }, SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
