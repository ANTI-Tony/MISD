"""
MISD Pipeline orchestrator — applies Phase 4 (safety) + Phase 5 (sympy) as a filter
on top of an already-optimized instruction set (e.g., Simple Opt output).

Inputs:
  --raw      JSON list of {"id", "instruction"}                  (original 5000)
  --opt      JSON list of {"id", "instruction"}                  (Simple Opt outputs)
  --typed    JSON list with "misd_type" field (from misd_phase1) (optional, for s4 hint)
  --raw_sol  JSON {"id": solver_output_text, ...}                (optional, for Phase 5)
  --opt_sol  JSON {"id": solver_output_text, ...}                (optional, for Phase 5)

Outputs:
  --output   JSON with per-sample verdict, kept/reverted instruction, and stats

Decision rule:
  - Phase 4 violation -> revert to raw
  - Phase 5 violation (when solver outputs available) -> drop sample entirely (default)
                                                       OR revert to raw (--p5-revert)

Stats printed at end.
"""
import argparse
import json
import os
import sys
from collections import Counter

# Local module imports (assume same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from misd_phase4_safety import safety_check
from misd_phase5_sympy import verify_pair


def load_id_map(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(x.get("id", i)): x for i, x in enumerate(data)}
    return {str(k): v for k, v in data.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--opt", required=True)
    p.add_argument("--typed", default=None)
    p.add_argument("--raw_sol", default=None)
    p.add_argument("--opt_sol", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--p5-revert", action="store_true",
                   help="On Phase 5 mismatch, revert to raw instead of dropping.")
    args = p.parse_args()

    raw_map = load_id_map(args.raw)
    opt_map = load_id_map(args.opt)
    typed_map = load_id_map(args.typed) if args.typed else {}
    raw_sol = json.load(open(args.raw_sol)) if args.raw_sol else {}
    opt_sol = json.load(open(args.opt_sol)) if args.opt_sol else {}

    common_ids = sorted(set(raw_map) & set(opt_map))
    print(f"Pairs: {len(common_ids)}")

    results = []
    p4_stats = Counter()
    p5_stats = Counter()
    final_action = Counter()

    for sid in common_ids:
        raw_ins = raw_map[sid]["instruction"]
        opt_ins = opt_map[sid]["instruction"]
        type_info = typed_map.get(sid, {}).get("misd_type") if typed_map else None

        # --- Phase 4: safety gate ---
        verdict, detail = safety_check(raw_ins, opt_ins, type_info=type_info)
        p4_stats[verdict] += 1

        if verdict != "kept":
            # Revert to raw; skip Phase 5 (raw == raw, trivially equivalent)
            final_ins = raw_ins
            final_action["p4_revert"] += 1
            results.append({
                "id": sid, "raw": raw_ins, "opt": opt_ins,
                "p4_verdict": verdict, "p4_detail": detail,
                "p5_verdict": "skipped",
                "final_action": "p4_revert", "final_instruction": final_ins,
            })
            continue

        # --- Phase 5: sympy answer-equivalence (only if solver outputs supplied) ---
        if args.raw_sol and args.opt_sol and sid in raw_sol and sid in opt_sol:
            eq, p5_detail = verify_pair(raw_sol[sid], opt_sol[sid])
            if eq is None:
                p5_stats["both_no_boxed"] += 1
                # Cannot verify; keep opt (Phase 4 said safe)
                final_ins = opt_ins
                final_action["p4_kept_p5_unverified"] += 1
            elif eq:
                p5_stats["equivalent"] += 1
                final_ins = opt_ins
                final_action["kept"] += 1
            else:
                p5_stats["mismatch"] += 1
                if args.p5_revert:
                    final_ins = raw_ins
                    final_action["p5_revert"] += 1
                else:
                    # Drop entirely (don't emit final_instruction)
                    final_action["p5_drop"] += 1
                    results.append({
                        "id": sid, "raw": raw_ins, "opt": opt_ins,
                        "p4_verdict": "kept", "p5_verdict": "mismatch",
                        "p5_detail": p5_detail,
                        "final_action": "p5_drop", "final_instruction": None,
                    })
                    continue
            results.append({
                "id": sid, "raw": raw_ins, "opt": opt_ins,
                "p4_verdict": "kept", "p5_verdict": "equivalent" if eq else ("none" if eq is None else "mismatch"),
                "p5_detail": p5_detail,
                "final_action": list(final_action)[-1] if final_action else "kept",
                "final_instruction": final_ins,
            })
        else:
            # No Phase 5 info available; keep opt
            final_ins = opt_ins
            final_action["p4_kept_no_p5"] += 1
            results.append({
                "id": sid, "raw": raw_ins, "opt": opt_ins,
                "p4_verdict": "kept", "p5_verdict": "skipped_no_solver",
                "final_action": "p4_kept_no_p5", "final_instruction": final_ins,
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== Phase 4 verdicts ===")
    for k, v in sorted(p4_stats.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:5d}  ({100*v/len(common_ids):.1f}%)")
    print("\n=== Phase 5 verdicts ===")
    for k, v in sorted(p5_stats.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:5d}")
    print("\n=== Final action ===")
    for k, v in sorted(final_action.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v:5d}  ({100*v/len(common_ids):.1f}%)")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
