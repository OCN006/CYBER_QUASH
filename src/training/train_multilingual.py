import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import numpy as np
import torch.nn.functional as F
import os

DATA_PATH = "data/processed/multilingual_clean.csv"
MODEL_SAVE = "models/xlm_roberta_multilingual"

class MultiLingualDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train():
    print("📥 Loading multilingual dataset...")
    df = pd.read_csv(DATA_PATH)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")

    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=3)

    train_ds = MultiLingualDataset(train_df, tokenizer)
    val_ds = MultiLingualDataset(val_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model.to(device)

    EPOCHS = 2

    for epoch in range(EPOCHS):
        print(f"\n====== EPOCH {epoch+1}/{EPOCHS} ======")

        # TRAIN
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print("Train Loss:", avg_train_loss)

        # VALIDATION
        model.eval()
        total_val_loss = 0
        correct, total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss
                total_val_loss += loss.item()

                preds = torch.argmax(out.logits, dim=1)
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = correct / total

        print("Val Loss:", avg_val_loss)
        print("Val Accuracy:", round(accuracy, 4))

    # SAVE MODEL
    print("💾 Saving model...")
    os.makedirs(MODEL_SAVE, exist_ok=True)
    model.save_pretrained(MODEL_SAVE)
    tokenizer.save_pretrained(MODEL_SAVE)

    print("🎉 Training complete! Model saved at:", MODEL_SAVE)

if __name__ == "__main__":
    train()
