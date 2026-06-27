"""
MISD Phase 5: LLM solver wrapper for sympy answer-equivalence verification.

For each (raw, opt) instruction pair:
  1. Call LLM on raw instruction -> raw_solution
  2. Call LLM on opt instruction -> opt_solution
  3. Extract \\boxed{} from each
  4. Verify symbolic equivalence via misd_phase5_sympy.answers_equivalent

Output JSON: per-pair verdict (equivalent / mismatch / both_no_boxed / one_no_boxed).

Usage:
  python misd_phase5_solve.py \
    --pairs data/optimized/misd_filter_fir_v2.json \
    --output data/optimized/misd_phase5_fir_v2.json
"""
import argparse
import asyncio
import json
import os
import sys
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misd_phase5_sympy import extract_boxed, answers_equivalent

EXPERT_URL = os.environ.get("EXPERT_URL", "http://localhost:8000/v1")
EXPERT_MODEL = os.environ.get("EXPERT_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SOLVE_PROMPT = """Solve the following math problem step by step. Give the final answer inside \\boxed{{}}.

Problem: {problem}

Solution:"""


async def solve_one(client, problem, sem, model):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": SOLVE_PROMPT.format(problem=problem)}],
                temperature=0.0,
                max_tokens=1024,
            )
            return r.choices[0].message.content
        except Exception as e:
            return f"[ERROR: {e}]"


async def verify_pair(client, sem, model, pair):
    raw_ins, opt_ins = pair["raw"], pair["opt"]
    if raw_ins == opt_ins:
        return {**pair, "p5_verdict": "identical_skipped"}
    raw_sol, opt_sol = await asyncio.gather(
        solve_one(client, raw_ins, sem, model),
        solve_one(client, opt_ins, sem, model),
    )
    a_raw = extract_boxed(raw_sol)
    a_opt = extract_boxed(opt_sol)
    if a_raw is None and a_opt is None:
        verdict, detail = "both_no_boxed", None
    elif a_raw is None or a_opt is None:
        verdict, detail = "one_no_boxed", f"raw={a_raw} opt={a_opt}"
    else:
        eq, reason = answers_equivalent(a_raw, a_opt)
        verdict = "equivalent" if eq else "mismatch"
        detail = f"{reason} | raw={a_raw!r} opt={a_opt!r}"
    return {
        **pair, "p5_verdict": verdict, "p5_detail": detail,
        "raw_answer": a_raw, "opt_answer": a_opt,
        "raw_solution_excerpt": (raw_sol or "")[-300:],
        "opt_solution_excerpt": (opt_sol or "")[-300:],
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", required=True,
                   help="JSON list with 'raw' and 'opt' fields per record (e.g., misd_pipeline output)")
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="0=all")
    p.add_argument("--only-kept", action="store_true",
                   help="Only verify Phase 4-kept pairs (skip already-reverted)")
    args = p.parse_args()

    with open(args.pairs, "r", encoding="utf-8") as f:
        data = json.load(f)
    if args.only_kept:
        data = [r for r in data if r.get("p4_verdict") == "kept"]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"Verifying {len(data)} pairs (model={EXPERT_MODEL})")

    client = AsyncOpenAI(base_url=EXPERT_URL, api_key="not-needed", timeout=180.0, max_retries=0)
    sem = asyncio.Semaphore(args.workers)

    tasks = [verify_pair(client, sem, EXPERT_MODEL, r) for r in data]
    results = []
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Phase 5"):
        results.append(await coro)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    from collections import Counter
    c = Counter(r["p5_verdict"] for r in results)
    print(f"\n=== MISD Phase 5 Stats ===")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:5d}  ({100*v/len(results):.1f}%)")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
