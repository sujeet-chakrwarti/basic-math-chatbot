# app.py
# Streamlit entry point.
# Run:  streamlit run app.py

import os
import streamlit as st

from config import APP_TITLE, APP_TAGLINE, MODEL_PATH
from chatbot.solver import solve
from chatbot.intent_model import IntentPredictor


# ── Auto-train if model doesn't exist ───────────────────────

def _ensure_model() -> None:
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⚙️ First launch — training intent model (takes ~30 seconds)…"):
            from chatbot.train_model import train
            train(silent=True)


# ── Load model (cached so it loads only once) ───────────────

@st.cache_resource(show_spinner="Loading model…")
def _load_predictor() -> IntentPredictor:
    _ensure_model()
    return IntentPredictor()


# ── Intent → friendly display name ──────────────────────────

_INTENT_LABELS = {
    "addition":        "➕ Addition",
    "subtraction":     "➖ Subtraction",
    "multiplication":  "✖️ Multiplication",
    "division":        "➗ Division",
    "percentage":      "💯 Percentage",
    "lcm":             "🔢 LCM",
    "hcf":             "🔢 HCF",
    "trigonometry":    "📐 Trigonometry",
    "statistics":      "📊 Statistics",
    "probability":     "🎲 Probability",
    "height_distance": "📏 Height & Distance",
    "polynomial":      "🔣 Polynomial Roots",
    "greeting":        "👋 Greeting",
}

# ── Example questions shown in the sidebar ──────────────────

_EXAMPLES = [
    ("➕ Addition",           "add 125 and 375"),
    ("➖ Subtraction",        "what is 500 minus 237"),
    ("✖️ Multiplication",     "multiply 24 by 15"),
    ("➗ Division",           "divide 144 by 12"),
    ("💯 Percentage",         "10.5% of 200"),
    ("💯 % Word Problem",     "has 10,025 cows sells 16% how many left"),
    ("🔢 LCM",                "lcm of 12 and 18"),
    ("🔢 HCF",                "hcf of 48 and 36"),
    ("📐 Trigonometry",       "sin 30 degrees"),
    ("📊 Mean",               "mean of 4 8 12 16 20"),
    ("📊 Median",             "median of 3 7 1 9 5"),
    ("📊 Mode",               "mode of 2 2 3 3 3 4"),
    ("🎲 Probability",        "probability of rolling 6 on a die"),
    ("📏 Height & Distance",  "angle of elevation 45 degrees distance 100 m"),
    ("🔣 Polynomial",         "find roots of x^2 - 5x + 6"),
]


# ════════════════════════════════════════════════════════════
# Page setup
# ════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MathBot",
    page_icon="🧮",
    layout="centered",
)

st.title(APP_TITLE)
st.caption(APP_TAGLINE)

# ── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    st.header("💡 Example Questions")
    st.markdown("Click any example to send it:")
    for label, question in _EXAMPLES:
        if st.button(f"{label}", key=question, use_container_width=True):
            st.session_state["pending_input"] = question

    st.divider()
    st.markdown("**Supported Topics**")
    st.markdown(
        "- Basic Arithmetic\n"
        "- Percentage\n"
        "- LCM & HCF\n"
        "- Trigonometry\n"
        "- Mean · Median · Mode\n"
        "- Probability\n"
        "- Height & Distance\n"
        "- Polynomial Roots"
    )
    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state["messages"] = []


# ── Session state ────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "pending_input" not in st.session_state:
    st.session_state["pending_input"] = ""


# ── Load model ───────────────────────────────────────────────

predictor = _load_predictor()


# ── Render existing chat history ─────────────────────────────

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Handle new input ─────────────────────────────────────────

def _process(user_input: str) -> None:
    """Classify intent, solve, and append messages to session state."""
    user_input = user_input.strip()
    if not user_input:
        return

    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Classify
    intent, confidence = predictor.predict(user_input)
    label = _INTENT_LABELS.get(intent, intent)

    # ── Out-of-scope guard ───────────────────────────────────
    # If confidence is low AND intent isn't greeting, the question
    # is probably not a math topic we support.
    if confidence < 0.50 and intent != "greeting":
        answer = (
            "🤔 **I\'m not sure how to help with that.**\n\n"
            "I\'m a math-focused chatbot. I can help with:\n\n"
            "- ➕ Basic Arithmetic\n"
            "- 💯 Percentage\n"
            "- 🔢 LCM & HCF\n"
            "- 📐 Trigonometry\n"
            "- 📊 Mean · Median · Mode\n"
            "- 🎲 Probability\n"
            "- 📏 Height & Distance\n"
            "- 🔣 Polynomial Roots\n\n"
            "Try asking something like *sin 30 degrees* or *find roots of x² - 5x + 6*."
        )
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        return

    # Solve
    with st.chat_message("assistant"):
        with st.spinner("Solving…"):
            try:
                answer = solve(intent, user_input)
            except Exception as exc:
                answer = (
                    f"❌ Could not solve this problem.\n\n"
                    f"**Detected topic :** {label}\n\n"
                    f"Please rephrase your question or check the examples in the sidebar."
                )

        # Show confidence badge
        badge_color = "green" if confidence >= 0.85 else "orange" if confidence >= 0.6 else "red"
        st.markdown(
            f"<small>🏷️ Topic: <b>{label}</b> &nbsp;|&nbsp; "
            f"Confidence: <span style='color:{badge_color}'><b>{confidence:.0%}</b></span></small>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})


# ── Input sources ─────────────────────────────────────────────

# 1. Sidebar example button
if st.session_state["pending_input"]:
    pending = st.session_state.pop("pending_input")
    _process(pending)
    st.rerun()

# 2. Chat input box
user_text = st.chat_input("Ask a math question…")
if user_text:
    _process(user_text)