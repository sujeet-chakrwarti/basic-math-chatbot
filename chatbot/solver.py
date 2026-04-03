# chatbot/solver.py
# Step-by-step math solvers for all supported intents.
# Libraries: math · statistics · fractions · sympy · numpy

from __future__ import annotations
import re
import math
import statistics
from fractions import Fraction

import numpy as np
import sympy as sp


# ══════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════

def _clean(text: str) -> str:
    """Strip comma thousands-separators and OCR % artifacts."""
    s = text
    while re.search(r"\d,\d{3}(?!\d)", s):
        s = re.sub(r"(\d),(\d{3})(?!\d)", r"\1\2", s)
    s = re.sub(r"(%)\d+\b", r"\1", s)   # "16%6" → "16%"
    return s


def _nums(text: str) -> list[float]:
    """Extract all numbers from text (handles comma-thousands)."""
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", _clean(text))]


def _fmt(n: float) -> str:
    """Show as integer if whole, else as decimal."""
    return str(int(n)) if n == int(n) else f"{n:g}"


def _sep(title: str) -> str:
    return f"{'─' * 48}\n📘 {title}\n{'─' * 48}"


# ══════════════════════════════════════════════════════════════
# Basic Arithmetic
# ══════════════════════════════════════════════════════════════

def solve_addition(text: str) -> str:
    nums = _nums(text)
    if len(nums) < 2:
        return "❌ Please give at least two numbers.\n   e.g. *add 25 and 47*"
    total = sum(nums)
    expr  = " + ".join(_fmt(n) for n in nums)
    return (
        f"{_sep('Addition')}\n"
        f"**Expression :** {expr}\n\n"
        f"**Step 1 :** Add all numbers together\n"
        f"**Result  :** {expr} = **{_fmt(total)}**\n\n"
        f"✅ **Answer : {_fmt(total)}**"
    )


def solve_subtraction(text: str) -> str:
    nums = _nums(text)
    if len(nums) < 2:
        return "❌ Please give at least two numbers.\n   e.g. *100 minus 37*"
    result = nums[0]
    for n in nums[1:]:
        result -= n
    expr = " − ".join(_fmt(n) for n in nums)
    return (
        f"{_sep('Subtraction')}\n"
        f"**Expression :** {expr}\n\n"
        f"**Step 1 :** Subtract values left to right\n"
        f"**Result  :** {expr} = **{_fmt(result)}**\n\n"
        f"✅ **Answer : {_fmt(result)}**"
    )


def solve_multiplication(text: str) -> str:
    nums = _nums(text)
    if len(nums) < 2:
        return "❌ Please give at least two numbers.\n   e.g. *multiply 6 by 9*"
    result = nums[0]
    for n in nums[1:]:
        result *= n
    expr = " × ".join(_fmt(n) for n in nums)
    return (
        f"{_sep('Multiplication')}\n"
        f"**Expression :** {expr}\n\n"
        f"**Step 1 :** Multiply all values together\n"
        f"**Result  :** {expr} = **{_fmt(result)}**\n\n"
        f"✅ **Answer : {_fmt(result)}**"
    )


def solve_division(text: str) -> str:
    nums = _nums(text)
    if len(nums) < 2:
        return "❌ Please give two numbers.\n   e.g. *divide 100 by 4*"
    if nums[1] == 0:
        return "❌ Division by zero is undefined."
    quotient  = int(nums[0] // nums[1])
    remainder = int(nums[0] % nums[1])
    result    = nums[0] / nums[1]
    return (
        f"{_sep('Division')}\n"
        f"**Expression :** {_fmt(nums[0])} ÷ {_fmt(nums[1])}\n\n"
        f"**Step 1 :** Divide {_fmt(nums[0])} by {_fmt(nums[1])}\n"
        f"**Step 2 :** Integer quotient = {quotient},  remainder = {remainder}\n\n"
        f"✅ **Answer : {result:g}**"
    )


# ══════════════════════════════════════════════════════════════
# Percentage
# ══════════════════════════════════════════════════════════════

def _pct_val(text: str) -> float | None:
    """Extract the percentage value from text."""
    cleaned = re.sub(r"(%)\d+\b", r"\1", text)
    m = re.search(r"(\d+\.?\d*)\s*%", cleaned)
    if m:
        return float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s+percent(?:age)?", text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _exclude_num(text: str, exclude: float) -> list[float]:
    result, removed = [], False
    for n in _nums(text):
        if not removed and abs(n - exclude) < 1e-9:
            removed = True
            continue
        result.append(n)
    return result


_REMOVE_KW = {"sell","sold","sells","selling","spent","spend","donated","give","gave",
              "lost","lose","used","removed","eaten","ate","taken","withdrew","paid"}
_ADD_KW    = {"buy","bought","added","gained","received","earned","deposited","increased"}
_REMAIN_KW = {"left","remain","remaining","still","rest","leftover","balance","have","has"}


def solve_percentage(text: str) -> str:
    lower    = text.lower()
    all_nums = _nums(text)
    pct      = _pct_val(text)
    words    = set(re.findall(r"\b\w+\b", lower))

    # ── "what percentage is X of Y?" ────────────────────────
    if re.search(r"what\s+percent|percent(?:age)?\s+is\b|how\s+many\s+percent|how\s+much\s+percent", lower):
        if len(all_nums) < 2:
            return "❌ Provide two numbers, e.g. *what percentage is 25 of 100*"
        part, whole = all_nums[0], all_nums[1]
        if whole == 0:
            return "❌ The whole value cannot be zero."
        result = (part / whole) * 100
        return (
            f"{_sep('Percentage — Find %')}\n"
            f"**Part  :** {_fmt(part)}\n"
            f"**Whole :** {_fmt(whole)}\n\n"
            f"**Formula :** (Part ÷ Whole) × 100\n"
            f"**Step 1  :** ({_fmt(part)} ÷ {_fmt(whole)}) × 100\n\n"
            f"✅ **Answer : {result:.4g}%**"
        )

    # ── Percent change ───────────────────────────────────────
    if re.search(r"\b(increase|decrease|change|grew|growth|dropped|fell|rose)\b", lower):
        if len(all_nums) < 2:
            return "❌ Provide two values, e.g. *percent increase from 80 to 100*"
        old_v, new_v = all_nums[0], all_nums[1]
        if old_v == 0:
            return "❌ Original value cannot be zero."
        change    = ((new_v - old_v) / old_v) * 100
        direction = "increase" if change >= 0 else "decrease"
        return (
            f"{_sep('Percent Change')}\n"
            f"**Old value :** {_fmt(old_v)}\n"
            f"**New value :** {_fmt(new_v)}\n\n"
            f"**Formula :** ((New − Old) ÷ Old) × 100\n"
            f"**Step 1  :** (({_fmt(new_v)} − {_fmt(old_v)}) ÷ {_fmt(old_v)}) × 100\n\n"
            f"✅ **Answer : {abs(change):.4g}% {direction}**"
        )

    # ── Word problem ("has 1000 items, sells 20%, how many left?") ──
    if pct is not None and (words & _REMOVE_KW or words & _ADD_KW or words & _REMAIN_KW):
        others = _exclude_num(text, pct)
        total  = max(others) if others else None
        if total is not None:
            amount = (pct / 100) * total
            if words & _REMOVE_KW or words & _REMAIN_KW:
                remaining = total - amount
                return (
                    f"{_sep('Percentage — Word Problem')}\n"
                    f"**Total    :** {_fmt(total)}\n"
                    f"**Removed  :** {pct}%\n\n"
                    f"**Step 1 — Find amount removed:**\n"
                    f"   ({pct} ÷ 100) × {_fmt(total)} = {amount:g}\n\n"
                    f"**Step 2 — Subtract:**\n"
                    f"   {_fmt(total)} − {amount:g} = {remaining:g}\n\n"
                    f"✅ **Answer : {remaining:g} remaining**"
                )
            else:
                new_total = total + amount
                return (
                    f"{_sep('Percentage — Word Problem')}\n"
                    f"**Original :** {_fmt(total)}\n"
                    f"**Added    :** {pct}%\n\n"
                    f"**Step 1 — Find amount added:**\n"
                    f"   ({pct} ÷ 100) × {_fmt(total)} = {amount:g}\n\n"
                    f"**Step 2 — Add:**\n"
                    f"   {_fmt(total)} + {amount:g} = {new_total:g}\n\n"
                    f"✅ **Answer : {new_total:g}**"
                )

    # ── Simple "X% of Y" ─────────────────────────────────────
    if pct is not None:
        others = _exclude_num(text, pct)
        if others:
            total  = others[0]
            result = (pct / 100) * total
            return (
                f"{_sep('Percentage')}\n"
                f"**Formula :** (Percentage ÷ 100) × Total\n\n"
                f"**Step 1  :** ({pct} ÷ 100) × {_fmt(total)}\n\n"
                f"✅ **Answer : {result:g}**"
            )

    return (
        "❌ Could not parse this percentage problem.\n\n"
        "**Examples:**\n"
        "- *20% of 80*\n"
        "- *what percentage is 25 of 100*\n"
        "- *percent increase from 80 to 100*\n"
        "- *has 500 items, sells 30%, how many left?*"
    )


# ══════════════════════════════════════════════════════════════
# LCM
# ══════════════════════════════════════════════════════════════

def solve_lcm(text: str) -> str:
    nums = [int(n) for n in _nums(text) if n > 0]
    if len(nums) < 2:
        return "❌ Provide at least two positive integers.\n   e.g. *lcm of 12 and 18*"
    result = nums[0]
    steps  = []
    for n in nums[1:]:
        prev   = result
        result = math.lcm(result, n)
        steps.append(f"LCM({prev}, {n}) = {result}")
    return (
        f"{_sep('LCM — Least Common Multiple')}\n"
        f"**Numbers :** {nums}\n\n"
        f"**Formula :** LCM(a, b) = |a × b| ÷ GCD(a, b)\n\n"
        f"**Steps   :** {' → '.join(steps)}\n\n"
        f"✅ **Answer : {result}**"
    )


# ══════════════════════════════════════════════════════════════
# HCF / GCD
# ══════════════════════════════════════════════════════════════

def solve_hcf(text: str) -> str:
    nums = [int(n) for n in _nums(text) if n > 0]
    if len(nums) < 2:
        return "❌ Provide at least two positive integers.\n   e.g. *hcf of 48 and 36*"
    result = nums[0]
    for n in nums[1:]:
        result = math.gcd(result, n)

    def prime_factors(n: int) -> str:
        factors, d = [], 2
        temp = n
        while d * d <= temp:
            while temp % d == 0:
                factors.append(d)
                temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        return " × ".join(map(str, factors)) if factors else str(n)

    factor_lines = "\n".join(f"   {n} = {prime_factors(n)}" for n in nums)
    return (
        f"{_sep('HCF — Highest Common Factor')}\n"
        f"**Numbers :** {nums}\n\n"
        f"**Prime Factorisations :**\n{factor_lines}\n\n"
        f"**Method :** Euclidean algorithm / common prime factors\n\n"
        f"✅ **Answer : {result}**"
    )


# ══════════════════════════════════════════════════════════════
# Trigonometry
# ══════════════════════════════════════════════════════════════

_EXACT_VALUES = {
    (0,  "sin"): "0",       (0,  "cos"): "1",       (0,  "tan"): "0",
    (30, "sin"): "1/2",     (30, "cos"): "√3/2",    (30, "tan"): "1/√3",
    (45, "sin"): "1/√2",    (45, "cos"): "1/√2",    (45, "tan"): "1",
    (60, "sin"): "√3/2",    (60, "cos"): "1/2",     (60, "tan"): "√3",
    (90, "sin"): "1",       (90, "cos"): "0",
}


def solve_trigonometry(text: str) -> str:
    nums  = _nums(text)
    lower = text.lower()

    if not nums:
        return "❌ Please provide an angle.\n   e.g. *sin 30 degrees*"

    angle_deg = nums[0]
    angle_rad = math.radians(angle_deg)

    if re.search(r"\bsin(?:e)?\b", lower):
        val   = math.sin(angle_rad)
        exact = _EXACT_VALUES.get((int(angle_deg), "sin"), "")
        return (
            f"{_sep(f'Trigonometry — sin({angle_deg}°)')}\n"
            f"**Step 1 :** Convert degrees to radians\n"
            f"   {angle_deg}° × (π/180) = {angle_rad:.6g} rad\n\n"
            f"**Step 2 :** Apply sin function\n"
            f"   sin({angle_rad:.4g}) = {val:.6g}\n\n"
            + (f"**Exact value :** {exact}\n\n" if exact else "")
            + f"✅ **Answer : {val:.6g}**"
        )

    if re.search(r"\bcos(?:ine)?\b", lower):
        val   = math.cos(angle_rad)
        exact = _EXACT_VALUES.get((int(angle_deg), "cos"), "")
        return (
            f"{_sep(f'Trigonometry — cos({angle_deg}°)')}\n"
            f"**Step 1 :** Convert degrees to radians\n"
            f"   {angle_deg}° × (π/180) = {angle_rad:.6g} rad\n\n"
            f"**Step 2 :** Apply cos function\n"
            f"   cos({angle_rad:.4g}) = {val:.6g}\n\n"
            + (f"**Exact value :** {exact}\n\n" if exact else "")
            + f"✅ **Answer : {val:.6g}**"
        )

    if re.search(r"\btan(?:gent)?\b", lower):
        if angle_deg % 180 == 90:
            return f"❌ tan({angle_deg}°) is **undefined** (vertical asymptote)."
        val   = math.tan(angle_rad)
        exact = _EXACT_VALUES.get((int(angle_deg), "tan"), "")
        return (
            f"{_sep(f'Trigonometry — tan({angle_deg}°)')}\n"
            f"**Step 1 :** Convert degrees to radians\n"
            f"   {angle_deg}° × (π/180) = {angle_rad:.6g} rad\n\n"
            f"**Step 2 :** tan θ = sin θ / cos θ\n"
            f"   tan({angle_rad:.4g}) = {val:.6g}\n\n"
            + (f"**Exact value :** {exact}\n\n" if exact else "")
            + f"✅ **Answer : {val:.6g}**"
        )

    return "❌ Please specify **sin**, **cos**, or **tan**.\n   e.g. *sin 30 degrees*"


# ══════════════════════════════════════════════════════════════
# Statistics — Mean, Median, Mode
# ══════════════════════════════════════════════════════════════

def solve_statistics(text: str) -> str:
    nums  = _nums(text)
    lower = text.lower()

    if not nums:
        return "❌ Please provide a list of numbers.\n   e.g. *mean of 2 4 6 8 10*"

    # ── Mode ─────────────────────────────────────────────────
    if "mode" in lower:
        from collections import Counter
        freq     = Counter(nums)
        max_freq = max(freq.values())
        modes    = sorted(k for k, v in freq.items() if v == max_freq)
        mode_str = ", ".join(_fmt(m) for m in modes)
        note     = " *(no unique mode — multimodal)*" if len(modes) > 1 else ""
        return (
            f"{_sep('Statistics — Mode')}\n"
            f"**Data :** {nums}\n\n"
            f"**Step 1 :** Count frequency of each value\n"
            f"   {dict(freq)}\n\n"
            f"**Step 2 :** Find most frequent value(s)\n"
            f"   {mode_str} appears {max_freq} time(s){note}\n\n"
            f"✅ **Answer : {mode_str}**"
        )

    # ── Median ───────────────────────────────────────────────
    if "median" in lower:
        s = sorted(nums)
        n = len(s)
        if n % 2 == 1:
            med      = s[n // 2]
            mid_note = f"Middle value = s[{n//2}] = {_fmt(med)}"
        else:
            l, r     = s[n//2 - 1], s[n//2]
            med      = (l + r) / 2
            mid_note = f"Average of middle two: ({_fmt(l)} + {_fmt(r)}) ÷ 2 = {med:g}"
        return (
            f"{_sep('Statistics — Median')}\n"
            f"**Data   :** {nums}\n\n"
            f"**Step 1 :** Sort → {s}\n\n"
            f"**Step 2 :** {mid_note}\n\n"
            f"✅ **Answer : {med:g}**"
        )

    # ── Mean / Average (default) ─────────────────────────────
    total    = sum(nums)
    mean_val = total / len(nums)
    expr     = " + ".join(_fmt(n) for n in nums)
    return (
        f"{_sep('Statistics — Mean (Average)')}\n"
        f"**Data :** {nums}\n\n"
        f"**Step 1 — Sum :**\n   {expr} = {_fmt(total)}\n\n"
        f"**Step 2 — Divide by count ({len(nums)}) :**\n"
        f"   {_fmt(total)} ÷ {len(nums)} = {mean_val:g}\n\n"
        f"✅ **Answer : {mean_val:g}**"
    )


# ══════════════════════════════════════════════════════════════
# Probability
# ══════════════════════════════════════════════════════════════

def solve_probability(text: str) -> str:
    nums  = _nums(text)
    lower = text.lower()

    # ── Coin ────────────────────────────────────────────────
    if re.search(r"\bcoin\b|\bheads\b|\btails\b", lower):
        return (
            f"{_sep('Probability — Coin Flip')}\n"
            f"**Total outcomes   :** 2  {{heads, tails}}\n"
            f"**Favourable       :** 1\n\n"
            f"**Formula :** P(event) = Favourable ÷ Total\n"
            f"**Step 1  :** P(heads) = 1 ÷ 2\n\n"
            f"✅ **Answer : 1/2 = 0.5 = 50%**"
        )

    # ── Dice ─────────────────────────────────────────────────
    if re.search(r"\bdi(?:e|ce)\b|\broll\b|\brolling\b", lower):
        if "even" in lower:
            return (
                f"{_sep('Probability — Even Number on Die')}\n"
                f"**Even faces :** {{2, 4, 6}} → 3 outcomes\n"
                f"**Total      :** 6\n\n"
                f"✅ **Answer : 3/6 = 1/2 = 0.5 = 50%**"
            )
        if "odd" in lower:
            return (
                f"{_sep('Probability — Odd Number on Die')}\n"
                f"**Odd faces :** {{1, 3, 5}} → 3 outcomes\n"
                f"**Total     :** 6\n\n"
                f"✅ **Answer : 3/6 = 1/2 = 0.5 = 50%**"
            )
        if nums:
            t = int(nums[0])
            if 1 <= t <= 6:
                return (
                    f"{_sep(f'Probability — Rolling {t} on a Die')}\n"
                    f"**Favourable :** 1  (only face {t})\n"
                    f"**Total      :** 6\n\n"
                    f"✅ **Answer : 1/6 ≈ {1/6:.4f} ≈ 16.67%**"
                )

    # ── Deck of cards ─────────────────────────────────────────
    if re.search(r"\bcards?\b|\bdeck\b|\bdraw(?:ing|n)?\b", lower):
        card_map = {
            "red":   ("26 red cards (hearts + diamonds)", "26/52 = 1/2 = 0.5 = 50%"),
            "black": ("26 black cards (clubs + spades)",  "26/52 = 1/2 = 0.5 = 50%"),
            "ace":   ("4 aces",  "4/52 = 1/13 ≈ 7.69%"),
            "king":  ("4 kings", "4/52 = 1/13 ≈ 7.69%"),
            "queen": ("4 queens","4/52 = 1/13 ≈ 7.69%"),
            "spade": ("13 spades","13/52 = 1/4 = 25%"),
            "heart": ("13 hearts","13/52 = 1/4 = 25%"),
        }
        for keyword, (fav, answer) in card_map.items():
            if keyword in lower:
                return (
                    f"{_sep(f'Probability — Drawing a {keyword.title()} Card')}\n"
                    f"**Favourable :** {fav}\n"
                    f"**Total      :** 52\n\n"
                    f"✅ **Answer : {answer}**"
                )

    # ── Generic  "X out of Y" or "X favourable, Y total" ────
    if len(nums) >= 2:
        fav, total = nums[0], nums[1]
        if total == 0:
            return "❌ Total outcomes cannot be zero."
        p = fav / total
        return (
            f"{_sep('Probability')}\n"
            f"**Favourable outcomes :** {_fmt(fav)}\n"
            f"**Total outcomes      :** {_fmt(total)}\n\n"
            f"**Formula :** P = Favourable ÷ Total\n"
            f"**Step 1  :** P = {_fmt(fav)} ÷ {_fmt(total)}\n\n"
            f"✅ **Answer : {p:.6g}  ({p*100:.4g}%)**"
        )

    return (
        "❌ Could not parse this probability problem.\n\n"
        "**Examples:**\n"
        "- *probability of heads in a coin flip*\n"
        "- *probability of rolling 6 on a die*\n"
        "- *probability 3 out of 10*\n"
        "- *probability of drawing an ace from a deck*"
    )


# ══════════════════════════════════════════════════════════════
# Height and Distance  (angle of elevation / depression)
# ══════════════════════════════════════════════════════════════

def solve_height_distance(text: str) -> str:
    """
    Solves standard height & distance problems using trigonometry.
    Patterns supported:
      - Angle of elevation + horizontal distance → height
      - Angle of depression + height → horizontal distance
      - Shadow length + angle → height of object
    """
    nums  = _nums(text)
    lower = text.lower()

    if len(nums) < 2:
        return (
            "❌ Please provide an angle and a distance/height.\n\n"
            "**Examples:**\n"
            "- *angle of elevation 30 degrees, horizontal distance 100 m*\n"
            "- *tower 50 m tall, angle of elevation 45 degrees, find distance*\n"
            "- *shadow 10 m, sun angle 60 degrees, find height of tree*"
        )

    # ── Elevation: find height from angle + horizontal dist ─
    if re.search(r"angle of elevation|elevation angle|looking up", lower):
        # Decide which number is angle and which is distance
        angle_deg = None
        distance  = None
        for n in nums:
            if 0 < n < 90 and angle_deg is None:
                angle_deg = n
            else:
                distance = n
        if angle_deg is None or distance is None:
            angle_deg, distance = nums[0], nums[1]

        # If a height is known, find horizontal distance
        if re.search(r"find.*distance|how far|horizontal distance", lower):
            height   = distance
            angle_r  = math.radians(angle_deg)
            horiz    = height / math.tan(angle_r)
            return (
                f"{_sep('Height & Distance — Angle of Elevation')}\n"
                f"**Known    :** Height = {_fmt(height)} m,  Angle = {angle_deg}°\n"
                f"**Find     :** Horizontal distance\n\n"
                f"**Formula  :** tan(θ) = Opposite / Adjacent\n"
                f"**Step 1   :** tan({angle_deg}°) = {math.tan(angle_r):.4f}\n"
                f"**Step 2   :** Distance = Height ÷ tan(θ)\n"
                f"              = {_fmt(height)} ÷ {math.tan(angle_r):.4f}\n\n"
                f"✅ **Answer : {horiz:.4g} m**"
            )

        angle_r = math.radians(angle_deg)
        height  = distance * math.tan(angle_r)
        return (
            f"{_sep('Height & Distance — Angle of Elevation')}\n"
            f"**Horizontal distance :** {_fmt(distance)} m\n"
            f"**Angle of elevation  :** {angle_deg}°\n\n"
            f"**Diagram :**\n"
            f"```\n"
            f"       * Top of object\n"
            f"      /|\n"
            f" h   / |\n"
            f"    /  |\n"
            f"   /θ  |\n"
            f"  *────*\n"
            f"  ←  d →\n"
            f"```\n\n"
            f"**Formula  :** tan(θ) = Height / Distance\n"
            f"**Step 1   :** tan({angle_deg}°) = {math.tan(angle_r):.6g}\n"
            f"**Step 2   :** Height = Distance × tan(θ)\n"
            f"              = {_fmt(distance)} × {math.tan(angle_r):.6g}\n\n"
            f"✅ **Answer : {height:.4g} m**"
        )

    # ── Depression: find horizontal distance from angle + height
    if re.search(r"angle of depression|depression angle|looking down", lower):
        angle_deg = nums[0]
        height    = nums[1]
        angle_r   = math.radians(angle_deg)
        distance  = height / math.tan(angle_r)
        return (
            f"{_sep('Height & Distance — Angle of Depression')}\n"
            f"**Height of observer  :** {_fmt(height)} m\n"
            f"**Angle of depression :** {angle_deg}°\n\n"
            f"**Note :** Angle of depression = Angle of elevation (alternate interior angles)\n\n"
            f"**Formula :** tan(θ) = Height / Horizontal Distance\n"
            f"**Step 1  :** tan({angle_deg}°) = {math.tan(angle_r):.6g}\n"
            f"**Step 2  :** Distance = Height ÷ tan(θ)\n"
            f"             = {_fmt(height)} ÷ {math.tan(angle_r):.6g}\n\n"
            f"✅ **Answer : {distance:.4g} m**"
        )

    # ── Shadow problem ────────────────────────────────────────
    if re.search(r"shadow|sun angle|sun elevation", lower):
        shadow_len = nums[0]
        angle_deg  = nums[1] if len(nums) > 1 else nums[0]
        angle_r    = math.radians(angle_deg)
        height     = shadow_len * math.tan(angle_r)
        return (
            f"{_sep('Height & Distance — Shadow Problem')}\n"
            f"**Shadow length :** {_fmt(shadow_len)} m\n"
            f"**Sun angle     :** {angle_deg}°\n\n"
            f"**Formula :** Height = Shadow × tan(Sun angle)\n"
            f"**Step 1  :** tan({angle_deg}°) = {math.tan(angle_r):.6g}\n"
            f"**Step 2  :** Height = {_fmt(shadow_len)} × {math.tan(angle_r):.6g}\n\n"
            f"✅ **Answer : {height:.4g} m**"
        )

    # ── Generic: angle + distance → height ───────────────────
    angle_deg = nums[0]
    distance  = nums[1]
    angle_r   = math.radians(angle_deg)
    height    = distance * math.tan(angle_r)
    return (
        f"{_sep('Height & Distance')}\n"
        f"**Angle    :** {angle_deg}°\n"
        f"**Distance :** {_fmt(distance)} m\n\n"
        f"**Formula :** Height = Distance × tan(Angle)\n"
        f"**Step 1  :** tan({angle_deg}°) = {math.tan(angle_r):.6g}\n"
        f"**Step 2  :** Height = {_fmt(distance)} × {math.tan(angle_r):.6g}\n\n"
        f"✅ **Answer : {height:.4g} m**"
    )



# ══════════════════════════════════════════════════════════════
# Greeting / small-talk
# ══════════════════════════════════════════════════════════════

_GOODBYE_WORDS = {"bye", "goodbye", "see you", "see ya", "later", "take care", "farewell"}
_THANKS_WORDS  = {"thanks", "thank you", "thank", "thx"}


def solve_greeting(text: str) -> str:
    lower = text.lower().strip()
    words = set(re.findall(r"\b\w+\b", lower))

    if words & _GOODBYE_WORDS:
        return (
            "👋 **Goodbye!**\n\n"
            "It was great helping you with math. Come back anytime!"
        )
    if words & _THANKS_WORDS:
        return (
            "😊 **You're welcome!**\n\n"
            "Happy to help. Feel free to ask another math question anytime."
        )
    if re.search(r"\b(who are you|what are you|what can you do|your name)\b", lower):
        return (
            "🧮 **I'm MathBot!**\n\n"
            "I solve step-by-step math problems. Topics I cover:\n\n"
            "- ➕ Basic Arithmetic\n"
            "- 💯 Percentage\n"
            "- 🔢 LCM & HCF\n"
            "- 📐 Trigonometry (sin, cos, tan)\n"
            "- 📊 Statistics (mean, median, mode)\n"
            "- 🎲 Probability\n"
            "- 📏 Height & Distance\n"
            "- 🔣 Polynomial Roots\n\n"
            "Just type a math question!"
        )
    if re.search(r"\b(how are you|how is it|how r u)\b", lower):
        return (
            "😄 **I'm doing great, thanks for asking!**\n\n"
            "Ready to solve math problems. What would you like to calculate?"
        )
    return (
        "👋 **Hello! Welcome to MathBot.**\n\n"
        "I can solve math problems step by step.\n\n"
        "**Try asking:**\n"
        "- *sin 30 degrees*\n"
        "- *find roots of x² - 5x + 6*\n"
        "- *mean of 4 8 12 16*\n"
        "- *20% of 80*\n\n"
        "What math problem can I help you with? 😊"
    )


# ══════════════════════════════════════════════════════════════
# Polynomial Roots
# ══════════════════════════════════════════════════════════════

def _parse_poly_expr(raw: str) -> str:
    """Convert human-written polynomial string to SymPy-parseable form."""
    s = raw.strip().lower()

    # Remove instruction noise (longest phrase first)
    for phrase in sorted([
        "find the roots of", "find roots of", "find the root of", "find root of",
        "find the zeros of", "find zeros of", "solve for x in", "solve for x",
        "roots of the polynomial", "zeros of the polynomial",
        "roots of polynomial", "roots of equation", "roots of",
        "zeros of", "quadratic roots", "quadratic", "polynomial",
        "solve", "factor",
    ], key=len, reverse=True):
        s = s.replace(phrase, "")

    s = re.sub(r"=\s*0\s*$", "", s).strip()

    # Factored form: (x-3)(x+2) → (x-3)*(x+2)
    s = re.sub(r"\)\s*\(", ")*(", s)

    # Unicode superscripts
    s = (s.replace("x²", "x**2").replace("x³", "x**3")
          .replace("x⁴", "x**4").replace("x⁵", "x**5"))

    # x^n → x**n
    s = re.sub(r"x\s*\^\s*(\d+)", r"x**\1", s)

    # NxM (e.g. 2x2, 3x3) → N*x**M  (must run before the next two steps)
    s = re.sub(r"(\d)x(\d+)", r"\1*x**\2", s)

    # x2, x3 (no operator after x) → x**2, x**3
    s = re.sub(r"\bx(\d+)\b", r"x**\1", s)

    # 2x, 3x → 2*x, 3*x
    s = re.sub(r"(\d)\s*\*?\s*x\b", r"\1*x", s)

    # 2(x+1) → 2*(x+1)
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)
    s = re.sub(r"\)\s*(\d)", r")*\1", s)

    return s.strip()


def _format_root(r: sp.Basic, idx: int) -> str:
    """Format one root: integer / p/q fraction / √ irrational / complex."""
    r_s = sp.simplify(r)

    def _nice(expr) -> str:
        s = str(expr)
        s = re.sub(r"sqrt\((\d+)\)", r"√\1", s)
        s = re.sub(r"sqrt\(([^)]+)\)", r"√(\1)", s)
        s = re.sub(r"\*I\b", "·i", s)
        s = re.sub(r"\bI\b",  "i",  s)
        s = s.replace("**", "^").replace("*", "·")
        return s

    try:
        c = complex(r_s)
        is_real = abs(c.imag) < 1e-9

        if is_real:
            v = c.real
            if abs(v - round(v)) < 1e-9:
                return f"  **x{idx}** = {int(round(v))}"
            frac = Fraction(v).limit_denominator(1000)
            if abs(float(frac) - v) < 1e-9 and frac.denominator != 1:
                return f"  **x{idx}** = {frac.numerator}/{frac.denominator} = {v:.6g}"
            return f"  **x{idx}** = {_nice(r_s)} ≈ {v:.6g}"
        else:
            tag = "*(imaginary)*" if abs(c.real) < 1e-9 else "*(complex)*"
            return f"  **x{idx}** = {_nice(r_s)}  {tag}"

    except (TypeError, ValueError):
        return f"  **x{idx}** = {_nice(r_s)}"


def solve_polynomial(text: str) -> str:
    """Find all roots of a polynomial. Supports standard, factored and cubic forms."""
    x = sp.Symbol("x")
    expr_str = _parse_poly_expr(text)

    if not expr_str:
        return (
            "❌ Could not identify a polynomial expression.\n\n"
            "**Examples:**\n"
            "- *find roots of x² - 5x + 6*\n"
            "- *solve 2x² + 3x - 2 = 0*\n"
            "- *roots of x³ - 6x² + 11x - 6*\n"
            "- *find roots of (x-3)(x-2)(x+1)*"
        )

    try:
        expr  = sp.expand(sp.sympify(expr_str, locals={"x": x}))
        roots = sp.solve(expr, x)
    except Exception:
        return _solve_quadratic_fallback(text)

    if not roots:
        return f"❌ No roots found for: `{expr_str}`"

    try:
        deg = sp.Poly(expr, x).degree()
    except Exception:
        deg = "?"

    # Discriminant note for quadratics
    disc_note = ""
    if deg == 2:
        try:
            coeffs = sp.Poly(expr, x).all_coeffs()
            if len(coeffs) == 3:
                a_c, b_c, c_c = [float(c) for c in coeffs]
                disc = b_c**2 - 4 * a_c * c_c
                nature = (
                    "two distinct real roots" if disc > 0
                    else "one repeated root (perfect square)" if disc == 0
                    else "two complex roots  (no real solutions)"
                )
                disc_note = (
                    f"**Quadratic formula :** x = (−b ± √(b²−4ac)) / 2a\n"
                    f"**Coefficients :** a = {a_c:g},  b = {b_c:g},  c = {c_c:g}\n"
                    f"**Discriminant :** b²−4ac = {disc:g}  →  {nature}\n\n"
                )
        except Exception:
            pass

    # Format equation (no sp.pretty — it produces multi-line superscripts)
    expr_display = str(expr).replace("**", "^").replace("*", "·")

    root_lines = "\n".join(_format_root(r, i) for i, r in enumerate(roots, 1))

    deg_str = str(deg)
    return (
        f"{_sep('Polynomial Roots  (degree ' + deg_str + ')')}\n"
        f"**Equation :** {expr_display} = 0\n\n"
        f"{disc_note}"
        f"**Roots :**\n"
        f"{root_lines}\n\n"
        f"✅ **{len(roots)} root(s) found**"
    )


def _solve_quadratic_fallback(text: str) -> str:
    """Last-resort: treat three raw numbers as a, b, c of ax²+bx+c=0."""
    nums = _nums(text)
    if len(nums) < 2:
        return (
            "❌ Could not parse the polynomial.\n\n"
            "**Examples:**\n"
            "- *find roots of x² - 5x + 6*\n"
            "- *solve 2x² + 3x - 2 = 0*\n"
            "- *roots of (x-3)(x-2)(x+1)*\n"
            "- *quadratic 1 -5 6*  (coefficients a b c)"
        )
    a, b = (nums[0], nums[1]) if len(nums) >= 2 else (1.0, nums[0])
    c    = nums[2] if len(nums) >= 3 else 0.0
    if a == 0:
        x_val = -c / b if b != 0 else None
        if x_val is None:
            return "❌ Not a valid polynomial."
        return f"{_sep('Linear Root')}\n**x = {x_val:g}**\n\n✅ **1 root found**"

    disc = b**2 - 4 * a * c
    if disc < 0:
        rp = -b / (2 * a)
        ip = math.sqrt(-disc) / (2 * a)
        return (
            f"{_sep('Quadratic Roots')}\n"
            f"**Equation :** {a:g}x² + {b:g}x + {c:g} = 0\n\n"
            f"**Discriminant :** {disc:g}  →  two complex roots\n\n"
            f"**Roots :**\n"
            f"  **x1** = {rp:.4g} + {ip:.4g}i  *(complex)*\n"
            f"  **x2** = {rp:.4g} − {ip:.4g}i  *(complex)*\n\n"
            f"✅ **2 root(s) found**"
        )
    x1 = (-b + math.sqrt(disc)) / (2 * a)
    x2 = (-b - math.sqrt(disc)) / (2 * a)
    lines = "\n".join([
        _format_root(sp.nsimplify(x1, rational=True), 1),
        _format_root(sp.nsimplify(x2, rational=True), 2),
    ])
    return (
        f"{_sep('Quadratic Roots')}\n"
        f"**Equation :** {a:g}x² + {b:g}x + {c:g} = 0\n\n"
        f"**Formula  :** x = (−b ± √(b²−4ac)) / 2a\n"
        f"**Discriminant :** {disc:g}\n\n"
        f"**Roots :**\n{lines}\n\n"
        f"✅ **2 root(s) found**"
    )


# ══════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════

_SOLVERS: dict[str, callable] = {
    "addition":        solve_addition,
    "subtraction":     solve_subtraction,
    "multiplication":  solve_multiplication,
    "division":        solve_division,
    "percentage":      solve_percentage,
    "lcm":             solve_lcm,
    "hcf":             solve_hcf,
    "trigonometry":    solve_trigonometry,
    "statistics":      solve_statistics,
    "probability":     solve_probability,
    "height_distance": solve_height_distance,
    "polynomial":      solve_polynomial,
    "greeting":        solve_greeting,
}


def solve(intent: str, text: str) -> str:
    """Route *text* to the correct solver based on *intent*."""
    fn = _SOLVERS.get(intent)
    if fn is None:
        raise ValueError(f"No solver for intent: '{intent}'")
    return fn(text)