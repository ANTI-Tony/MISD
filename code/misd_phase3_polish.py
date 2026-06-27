"""
MISD Phase 3: Math-Specific Targeted Polishing (5 additive rules)

Each rule has:
  - trigger: pure-Python predicate over (instruction, misd_type) -> bool
  - prompt:  LLM prompt for the additive edit
  - flag:    enable/disable for ablation

Pipeline per sample:
  1. For each enabled rule whose trigger fires:
       a. Call LLM with rule-specific additive-edit prompt.
       b. Immediately run Phase 4 safety check on (current, polished).
       c. If safe, accept polished as new current; else discard this rule.
  2. Output final polished instruction + rule trail.

This design ensures every accepted polish is provably additive-safe.
"""
import argparse
import asyncio
import json
import os
import re
import sys

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misd_phase4_safety import safety_check

EXPERT_URL = os.environ.get("EXPERT_URL", "http://localhost:8000/v1")
EXPERT_MODEL = os.environ.get("EXPERT_MODEL", "Qwen/Qwen2.5-7B-Instruct")


# =====================================================================
# Trigger predicates
# =====================================================================
ASY_LABEL_RE = re.compile(r'label\s*\(\s*"([^"]+)"', re.IGNORECASE)
UNIT_HINTS = ['speed', 'distance', 'meter', 'mile', 'km', 'foot', 'feet',
              '速度', '距离', '米', '公里', 'second', 'hour']
LATEX_LONG_RE = re.compile(r'\$\$?[^$]{60,}\$\$?')
INTEGER_HINTS = ['integer', 'positive', 'natural', '正整数', '自然数']


def t_r1_labels(ins, mtype):
    """R1: Visual labels mentioned in asy but absent from conditions text."""
    if not mtype or not mtype.get("visual", {}).get("present"):
        return False
    asy_blocks = re.findall(r'\[asy\](.*?)\[/asy\]', ins, re.DOTALL | re.IGNORECASE)
    if not asy_blocks:
        return False
    labels = set()
    for b in asy_blocks:
        labels.update(ASY_LABEL_RE.findall(b))
    if not labels:
        return False
    text_only = re.sub(r'\[asy\].*?\[/asy\]', '', ins, flags=re.DOTALL | re.IGNORECASE)
    missing = [L for L in labels if L not in text_only and len(L) <= 5]
    return len(missing) > 0


def t_r2_units(ins, mtype):
    """R2: contains physical-quantity hints but bare numbers (asy stripped)."""
    text = re.sub(r'\[asy\].*?\[/asy\]', '', ins, flags=re.DOTALL | re.IGNORECASE)
    low = text.lower()
    if not any(h in low for h in UNIT_HINTS):
        return False
    return bool(re.search(r'\b\d+\b(?!\s*(meter|mile|km|m|ft|second|hour|s|h))', text))


def t_r3_format(ins, mtype):
    """R3: Format absent and answer is non-unique form (compute/find)."""
    if mtype and mtype.get("format", {}).get("present"):
        return False
    if "\\boxed" in ins:
        return False
    qkind = (mtype or {}).get("question", {}).get("kind", "compute")
    return qkind in ("compute", "find")


def t_r4_boundary(ins, mtype):
    """R4: Variables present but no integer/positive type declaration."""
    if not (mtype and mtype.get("conditions", {}).get("has_variables")):
        return False
    low = ins.lower()
    return not any(h in low for h in INTEGER_HINTS)


def t_r5_long_latex(ins, mtype):
    """R5: At least one LaTeX expression > 60 chars."""
    return bool(LATEX_LONG_RE.search(ins))


RULES = [
    ("R1", t_r1_labels,     "label_completion"),
    ("R2", t_r2_units,      "unit_completion"),
    ("R3", t_r3_format,     "format_explicit"),
    ("R4", t_r4_boundary,   "boundary_explicit"),
    ("R5", t_r5_long_latex, "intermediate_var"),
]


# =====================================================================
# LLM additive prompts (per rule)
# =====================================================================
RULE_PROMPTS = {
    "R1": """You are polishing a math problem instruction. Apply ONLY this rule:
- If the asy figure contains labeled points (e.g., "A","B","C") that are NOT mentioned in the surrounding text, add a brief sentence introducing those labels (e.g., "Let A, B, C be the points shown in the figure.").

STRICT RULES:
- DO NOT change any existing words, numbers, LaTeX, or asy code.
- DO NOT remove anything.
- ONLY add new clarifying text.
- If no edit is needed, output the original instruction unchanged.

Original instruction:
{instruction}

Output ONLY the polished instruction text (no explanation, no markdown):""",

    "R2": """You are polishing a math problem instruction. Apply ONLY this rule:
- If a number appears without a unit but the context implies a physical quantity (speed, distance, time), add the unit in parentheses next to that number.

STRICT RULES:
- DO NOT change any existing words, numbers, LaTeX, or asy code.
- DO NOT remove anything. ONLY add unit annotations.
- If no edit is needed, output the original instruction unchanged.

Original instruction:
{instruction}

Output ONLY the polished instruction text:""",

    "R3": """You are polishing a math problem instruction. Apply ONLY this rule:
- Append a single sentence at the end requesting the answer be given inside \\boxed{{}} in simplest form.

STRICT RULES:
- DO NOT change any existing words, numbers, LaTeX, or asy code.
- DO NOT remove anything.
- ONLY append the new sentence at the very end.

Original instruction:
{instruction}

Output ONLY the polished instruction text:""",

    "R4": """You are polishing a math problem instruction. Apply ONLY this rule:
- If the variables (e.g., n, k) appear without explicit type declaration but the problem context implies they must be positive integers, add a brief clause stating so (e.g., "where n is a positive integer.").

STRICT RULES:
- DO NOT change any existing words, numbers, LaTeX, or asy code.
- DO NOT remove anything.
- ONLY add the type declaration clause.
- If unsure whether the variable is a positive integer, output unchanged.

Original instruction:
{instruction}

Output ONLY the polished instruction text:""",

    "R5": """You are polishing a math problem instruction. Apply ONLY this rule:
- If a single LaTeX expression is very long (>60 chars), introduce a named intermediate variable for a sub-expression to improve readability (e.g., "Let u = ...; then ...").

STRICT RULES:
- The original LaTeX expression MUST still appear in full somewhere in the output.
- DO NOT change any existing numbers or variable names.
- DO NOT remove anything from the original.
- ONLY add the "Let u = ..." sentence as additional context.

Original instruction:
{instruction}

Output ONLY the polished instruction text:""",
}


# =====================================================================
# LLM call
# =====================================================================
async def call_llm(client, prompt, max_retries=3):
    for i in range(max_retries):
        try:
            r = await client.chat.completions.create(
                model=EXPERT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            return r.choices[0].message.content
        except Exception:
            if i < max_retries - 1:
                await asyncio.sleep(1 * (i + 1))
            else:
                return None


def clean_output(text):
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


async def polish_one(client, sample, sem, idx, enabled_rules):
    async with sem:
        ins = sample["instruction"]
        mtype = sample.get("misd_type")
        current = ins
        trail = []

        for rid, trigger, name in RULES:
            if rid not in enabled_rules:
                continue
            if not trigger(current, mtype):
                continue
            prompt = RULE_PROMPTS[rid].format(instruction=current)
            raw = await call_llm(client, prompt)
            polished = clean_output(raw)
            if not polished or polished == current:
                trail.append({"rule": rid, "action": "noop"})
                continue
            verdict, detail = safety_check(current, polished, type_info=mtype)
            if verdict == "kept":
                current = polished
                trail.append({"rule": rid, "action": "applied"})
            else:
                trail.append({"rule": rid, "action": "rejected", "reason": verdict, "detail": detail})

        return idx, {**sample, "polished_instruction": current, "polish_trail": trail}


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="JSON list with 'instruction' (and optional 'misd_type')")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rules", default="R1,R2,R3,R4,R5",
                   help="Comma-separated subset of rules to enable")
    args = p.parse_args()

    enabled = set(r.strip() for r in args.rules.split(",") if r.strip())
    print(f"Enabled rules: {sorted(enabled)}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = AsyncOpenAI(base_url=EXPERT_URL, api_key="not-needed", timeout=120.0, max_retries=0)
    sem = asyncio.Semaphore(args.workers)

    print(f"Polishing {len(data)} samples (model={EXPERT_MODEL})")
    tasks = [polish_one(client, s, sem, i, enabled) for i, s in enumerate(data)]
    results = [None] * len(data)
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="MISD Phase 3"):
        idx, r = await coro
        results[idx] = r

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from collections import Counter
    rule_applied = Counter()
    rule_rejected = Counter()
    rule_noop = Counter()
    n_changed = 0
    for r in results:
        if r["polished_instruction"] != r["instruction"]:
            n_changed += 1
        for t in r.get("polish_trail", []):
            {"applied": rule_applied, "rejected": rule_rejected, "noop": rule_noop}[t["action"]][t["rule"]] += 1

    print(f"\n=== MISD Phase 3 Stats ===")
    print(f"Total: {len(results)}, changed: {n_changed} ({100*n_changed/len(results):.1f}%)")
    print(f"Rule applied: {dict(rule_applied)}")
    print(f"Rule rejected (Phase 4 violation): {dict(rule_rejected)}")
    print(f"Rule noop (LLM no edit): {dict(rule_noop)}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
