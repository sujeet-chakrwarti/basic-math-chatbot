# config.py
# Central configuration — edit this file to tune the chatbot.

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "intent_model.pth")
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.json")

# ── Model hyper-parameters ───────────────────────────────────
HIDDEN_SIZE   = 128
EPOCHS        = 300
LEARNING_RATE = 0.001
BATCH_SIZE    = 16

# ── Supported intent labels ──────────────────────────────────
INTENTS = [
    "addition",
    "subtraction",
    "multiplication",
    "division",
    "percentage",
    "lcm",
    "hcf",
    "trigonometry",
    "statistics",
    "probability",
    "height_distance",
    "polynomial",
    "greeting",
]

# ── App display ──────────────────────────────────────────────
APP_TITLE   = "🧮 MathBot"
APP_TAGLINE = "Your step-by-step math solver"