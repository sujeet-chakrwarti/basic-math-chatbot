# 🧮 MathBot — Step-by-Step Math Solver

A clean, Streamlit-powered math chatbot that solves problems step by step.
No translation. No OCR. No fallback AI. Just math.

---

## Supported Topics

| Topic | Examples |
|---|---|
| Basic Arithmetic | `add 25 and 47`, `divide 144 by 12` |
| Percentage | `20% of 80`, `percent increase from 80 to 100` |
| LCM | `lcm of 12 and 18` |
| HCF / GCD | `hcf of 48 and 36` |
| Trigonometry | `sin 30 degrees`, `cos 45 degrees` |
| Statistics | `mean of 2 4 6 8`, `median of 3 7 1 9` |
| Probability | `probability of rolling 6 on a die` |
| Height & Distance | `angle of elevation 45 degrees, distance 100 m` |
| Polynomial Roots | `find roots of x^2 - 5x + 6` |

---

## Project Structure

```
mathbot/
├── app.py                  # Streamlit UI — entry point
├── config.py               # All settings in one place
├── requirements.txt
├── README.md
├── .gitignore
├── chatbot/
│   ├── __init__.py
│   ├── solver.py           # All math solvers
│   ├── intent_model.py     # PyTorch classifier
│   └── train_model.py      # Training script
├── data/
│   └── dataset.json        # 500+ labelled training examples
└── models/
    └── .gitkeep            # Trained model saved here (auto-generated)
```

---

## Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/mathbot.git
cd mathbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Train the model manually
python chatbot/train_model.py

# 4. Run the app
streamlit run app.py
```

> **Note:** The intent model trains automatically on first launch if no
> `models/intent_model.pth` file is found. This takes about 30 seconds.

---

## Deploy on Streamlit Cloud

1. Push the repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, set **Main file** to `app.py`.
4. Click **Deploy**.

The model trains automatically on first deployment.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Web UI |
| `torch` | Intent classifier (PyTorch) |
| `sympy` | Polynomial root solving |
| `numpy` | Numerical computation |
| `scipy` | Scientific math support |
