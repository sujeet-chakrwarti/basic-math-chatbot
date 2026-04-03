# chatbot/intent_model.py
# PyTorch bag-of-words intent classifier with domain-aware signal tokens.

import re
import os
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HIDDEN_SIZE, MODEL_PATH


# ── Tokeniser ────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Convert raw text into a list of tokens.
    Injects special signal tokens so the classifier can distinguish
    intents that share common words (e.g. numbers appearing in both
    statistics and percentage questions).
    """
    t = text.lower()
    tokens = re.findall(r"\b\w+\b|%", t)

    # Percentage signals
    if "%" in t:
        tokens.append("__pct_symbol__")
    if re.search(r"\d+\.?\d*\s*%", t) or re.search(r"\d+\.?\d*\s+percent", t):
        tokens.append("__pct_word__")
    if re.search(r"\b(discount|tax|off|markup|increase|decrease|profit|loss)\b", t):
        tokens.append("__pct_context__")

    # Statistics signals
    if re.search(r"\b(mean|median|mode|average|avg|arithmetic)\b", t):
        tokens.append("__stat_kw__")
    if re.search(r"\b(data|dataset|values|list|set|numbers|scores)\b", t):
        tokens.append("__data_kw__")

    # Trigonometry signals
    if re.search(r"\b(sin|cos|tan|sine|cosine|tangent|degree|degrees|angle|radian)\b", t):
        tokens.append("__trig_kw__")

    # Height & distance signals
    if re.search(r"\b(height|distance|tower|tree|building|shadow|cliff|elevation|depression|angle of elevation|angle of depression|top|foot|base)\b", t):
        tokens.append("__hd_kw__")

    # Polynomial signals
    if re.search(r"\b(polynomial|roots?|quadratic|equation|factor|solve for x|x\^|x\s*=)\b", t):
        tokens.append("__poly_kw__")
    if re.search(r"x\s*\*\*?\s*2|x\s*\^?\s*2|x²", t):
        tokens.append("__poly_kw__")

    # Fraction literals → strong fraction/lcm/hcf hint
    if re.search(r"\d+\s*/\s*\d+", t):
        tokens.append("__fraction_literal__")

    # LCM / HCF signals
    if re.search(r"\b(lcm|lowest common multiple|least common multiple)\b", t):
        tokens.append("__lcm_kw__")
    if re.search(r"\b(hcf|gcd|highest common factor|greatest common divisor|common factor)\b", t):
        tokens.append("__hcf_kw__")

    # Probability signals
    if re.search(r"\b(probability|chance|likelihood|odds|favorable|favourable|outcome|event)\b", t):
        tokens.append("__prob_kw__")

    return tokens


# ── Vocabulary helpers ───────────────────────────────────────

def build_vocab(token_lists: list[list[str]]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for tokens in token_lists:
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    return vocab


def bag_of_words(tokens: list[str], vocab: dict[str, int]) -> np.ndarray:
    bow = np.zeros(len(vocab), dtype=np.float32)
    for tok in tokens:
        if tok in vocab:
            bow[vocab[tok]] = 1.0
    return bow


# ── Neural network ───────────────────────────────────────────

class IntentNet(nn.Module):
    """Three-layer feed-forward classifier."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Predictor ────────────────────────────────────────────────

class IntentPredictor:
    """Load a trained checkpoint and predict intent labels."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'.\n"
                "Run:  python chatbot/train_model.py"
            )
        ckpt             = torch.load(model_path, map_location="cpu")
        self.vocab       = ckpt["vocab"]
        self.intents     = ckpt["intents"]
        self.model       = IntentNet(len(self.vocab), HIDDEN_SIZE, len(self.intents))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    def predict(self, text: str) -> tuple[str, float]:
        """Return (intent_label, confidence) for the given text."""
        tokens = tokenize(text)
        bow    = bag_of_words(tokens, self.vocab)
        x      = torch.tensor(bow).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), dim=1)
        conf, idx = torch.max(probs, dim=1)
        return self.intents[idx.item()], round(conf.item(), 4)
