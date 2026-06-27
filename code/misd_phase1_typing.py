"""
MISD Phase 1: 6-Component Structural Typing
Classify each math instruction into 6 components: Setup / Conditions / Question / Format / Visual / MCQ.

Server-side script. Requires vLLM serving a 7B classifier on EXPERT_URL.

Output: data/optimized/iqd_low_misd_typed.json
"""
import json
import os
import re
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

LOW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "iqd_low.json")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "optimized", "iqd_low_misd_typed.json")
EXPERT_URL = os.environ.get("EXPERT_URL", "http://localhost:8000/v1")
EXPERT_MODEL = os.environ.get("EXPERT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

TYPE_PROMPT = """You are decomposing a math problem instruction into its 6 structural components.

### Instruction
{instruction}

### Task
Output a JSON object with these 6 components. For each, decide if it is PRESENT and (where applicable) extract its key content.

1. **setup**: Background / scenario context (e.g., "三角形 ABC 中", "Mary buys 3 apples"). present:bool, summary:string (≤30 chars).
2. **conditions**: Given conditions including LaTeX expressions, equations, inequalities. present:bool, latex_count:int (number of LaTeX expressions), has_variables:bool.
3. **question**: The solving target. text:string (≤60 chars), kind: one of "compute|prove|find|judge|count".
4. **format**: Answer format constraint (e.g., "保留两位小数", "as a common fraction", "in interval form"). present:bool, constraint:string (≤30 chars, "" if absent).
5. **visual**: asy code or geometric drawing. present:bool, role: one of "critical|decorative|none" (critical = asy contains data not in text, decorative = redundant illustration).
6. **mcq**: multiple-choice options (A)~(E). present:bool, options_in_asy:bool (true if options are inside [asy] block).

### Output Rules (STRICT)
Respond with ONLY one valid JSON object on a single line. No prose. No markdown fences.

{{"setup":{{"present":<bool>,"summary":"..."}},"conditions":{{"present":<bool>,"latex_count":<int>,"has_variables":<bool>}},"question":{{"text":"...","kind":"<compute|prove|find|judge|count>"}},"format":{{"present":<bool>,"constraint":"..."}},"visual":{{"present":<bool>,"role":"<critical|decorative|none>"}},"mcq":{{"present":<bool>,"options_in_asy":<bool>}}}}
"""


HEURISTIC_LATEX_RE = re.compile(r'\\(frac|sqrt|sum|int|prod|sin|cos|tan|log|ln|exp|pi|theta|alpha|beta|gamma)\b')
HEURISTIC_VAR_RE = re.compile(r'(?<![a-zA-Z\\])[a-zA-Z](?![a-zA-Z])')
HEURISTIC_MCQ_RE = re.compile(r'\([A-E]\)')
HEURISTIC_ASY_RE = re.compile(r'\[asy\]', re.IGNORECASE)


def heuristic_features(ins):
    """Cheap pre-features used to (a) gate LLM calls, (b) cross-check LLM output."""
    has_asy = bool(HEURISTIC_ASY_RE.search(ins))
    has_mcq = bool(HEURISTIC_MCQ_RE.search(ins))
    latex_count = len(HEURISTIC_LATEX_RE.findall(ins))
    vars_ = set(HEURISTIC_VAR_RE.findall(ins))
    return {
        "has_asy": has_asy, "has_mcq": has_mcq,
        "latex_count": latex_count, "var_count": len(vars_),
    }


def parse_json_strict(text):
    if not text:
        return None
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def empty_typing(ins, h):
    return {
        "setup": {"present": True, "summary": ""},
        "conditions": {"present": h["latex_count"] > 0 or h["var_count"] > 0,
                       "latex_count": h["latex_count"], "has_variables": h["var_count"] > 0},
        "question": {"text": "", "kind": "compute"},
        "format": {"present": False, "constraint": ""},
        "visual": {"present": h["has_asy"], "role": "none" if not h["has_asy"] else "decorative"},
        "mcq": {"present": h["has_mcq"], "options_in_asy": False},
    }


def validate_typing(t, h):
    """Force consistency between LLM output and heuristic features."""
    if not isinstance(t, dict):
        return None
    required = {"setup", "conditions", "question", "format", "visual", "mcq"}
    if not required.issubset(t.keys()):
        return None
    # Cross-check visual
    if h["has_asy"] and not t["visual"].get("present"):
        t["visual"]["present"] = True
        if t["visual"].get("role", "none") == "none":
            t["visual"]["role"] = "decorative"
    if not h["has_asy"]:
        t["visual"]["present"] = False
        t["visual"]["role"] = "none"
    # Cross-check MCQ
    if h["has_mcq"] and not t["mcq"].get("present"):
        t["mcq"]["present"] = True
    if not h["has_mcq"]:
        t["mcq"]["present"] = False
        t["mcq"]["options_in_asy"] = False
    # Cross-check conditions latex_count
    if "conditions" in t:
        t["conditions"]["latex_count"] = h["latex_count"]
    return t


async def call_expert(client, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=EXPERT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=400,
            )
            return resp.choices[0].message.content
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                return None


async def type_one(client, sample, sem, idx):
    async with sem:
        ins = sample["instruction"]
        h = heuristic_features(ins)

        # Heuristic skip: pure short text without any structure -> emit defaults
        if not h["has_asy"] and not h["has_mcq"] and h["latex_count"] == 0 and len(ins) < 150:
            t = empty_typing(ins, h)
            t["_method"] = "heuristic_skip"
            return idx, {**sample, "misd_type": t}

        prompt = TYPE_PROMPT.format(instruction=ins[:3000])
        raw = await call_expert(client, prompt)
        parsed = parse_json_strict(raw)
        validated = validate_typing(parsed, h) if parsed else None

        if validated is None:
            t = empty_typing(ins, h)
            t["_method"] = "heuristic_fallback"
        else:
            validated["_method"] = "llm"
            t = validated
        return idx, {**sample, "misd_type": t}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--data", type=str, default=LOW_DATA_PATH)
    parser.add_argument("--output", type=str, default=OUTPUT_PATH)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = AsyncOpenAI(base_url=EXPERT_URL, api_key="not-needed", timeout=60.0, max_retries=0)
    sem = asyncio.Semaphore(args.workers)

    print(f"Typing {len(data)} samples (model={EXPERT_MODEL}, workers={args.workers})")
    tasks = [type_one(client, s, sem, i) for i, s in enumerate(data)]
    results = [None] * len(data)
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="MISD Phase 1"):
        idx, r = await coro
        results[idx] = r

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Stats
    from collections import Counter
    methods = Counter(r["misd_type"]["_method"] for r in results)
    visual_role = Counter(r["misd_type"]["visual"]["role"] for r in results)
    mcq = Counter(r["misd_type"]["mcq"]["present"] for r in results)
    fmt = Counter(r["misd_type"]["format"]["present"] for r in results)
    qkind = Counter(r["misd_type"]["question"]["kind"] for r in results)

    print("\n=== MISD Phase 1 Stats ===")
    print(f"Total: {len(results)}")
    print(f"Method: {dict(methods)}")
    print(f"Visual role: {dict(visual_role)}")
    print(f"MCQ present: {dict(mcq)}")
    print(f"Format present: {dict(fmt)}")
    print(f"Question kind: {dict(qkind)}")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
