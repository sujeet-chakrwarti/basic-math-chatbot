# chatbot/train_model.py
# Train the intent classifier.
# Run from project root:  python chatbot/train_model.py

import json
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_PATH, MODEL_PATH, EPOCHS, LEARNING_RATE, HIDDEN_SIZE, BATCH_SIZE
from chatbot.intent_model import tokenize, build_vocab, bag_of_words, IntentNet


# ── Dataset wrapper ──────────────────────────────────────────

class IntentDataset(Dataset):
    def __init__(self, data: list[dict], vocab: dict, intents: list[str]):
        label_map    = {label: i for i, label in enumerate(intents)}
        self.samples = [
            (
                torch.tensor(bag_of_words(tokenize(item["text"]), vocab)),
                torch.tensor(label_map[item["intent"]], dtype=torch.long),
            )
            for item in data
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Training routine ─────────────────────────────────────────

def train(silent: bool = False) -> None:
    """
    Train the model and save a checkpoint to MODEL_PATH.
    Pass silent=True to suppress output (used by app.py auto-train).
    """
    def log(msg: str) -> None:
        if not silent:
            print(msg)

    log("=" * 52)
    log("  MathBot — Training Intent Classifier")
    log("=" * 52)

    # Load dataset
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    intents     = sorted({item["intent"] for item in data})
    token_lists = [tokenize(item["text"]) for item in data]
    vocab       = build_vocab(token_lists)

    log(f"  Samples    : {len(data)}")
    log(f"  Intents    : {len(intents)}")
    log(f"  Vocab size : {len(vocab)}")
    log("-" * 52)

    # Build model
    dataset   = IntentDataset(data, vocab, intents)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model     = IntentNet(len(vocab), HIDDEN_SIZE, len(intents))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for X, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if not silent and (epoch % 50 == 0 or epoch == 1):
            avg = total_loss / len(loader)
            lr  = scheduler.get_last_lr()[0]
            log(f"  Epoch {epoch:>4}/{EPOCHS}  loss={avg:.4f}  lr={lr:.5f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "vocab": vocab, "intents": intents},
        MODEL_PATH,
    )
    log(f"\n[✓] Model saved → {MODEL_PATH}\n")


if __name__ == "__main__":
    train()
