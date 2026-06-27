# MISD: A Controlled Study of Math Instruction-Data Refinement

**When Refining Math Instruction Data Hurts: A Controlled Study of Format Scaffolding and Verification Filtering**

This repo studies whether the standard math-SFT-data recipe — *structure → rewrite into a canonical boxed format → filter by symbolic answer-equivalence* — actually generalizes. Using a strictly controlled protocol (same base model, same LoRA config, same eval; **only the training data varies**), we decompose the pipeline into its individual operators and measure each on an honest **9-benchmark** suite: 5 in-distribution + 4 out-of-distribution (incl. Chinese **CMATH**).

## Key findings (all on Qwen2.5-3B, multi-seed where headline)

1. **The fully refined pipeline is the *worst* configuration on the honest suite** (61.5 nine-bench avg), *below* an unrefined Claude-rewrite baseline (63.7). In-distribution gains (+1.5) are erased by an OOD collapse driven by a cross-lingual format tax (CMATH −12.6).
2. **The boxed-answer prompt is the single largest *harmful* operator**: it helps in-distribution extraction (GSM8K +14) but destroys cross-lingual transfer (CMATH −30). The config with **neither** the boxed prompt nor rewrite rules reaches **72.6 ± 0.4** (3 seeds), **+8.9** over the strongest baseline.
3. **The symbolic-verification filter contributes ≈ nothing**: dropping the same number of *random* samples matches dropping the solver-flagged ones (72.1 vs 72.6).

**Takeaway:** for math SFT data, format normalization is double-edged and verification filtering is overrated; lighter-touch refinement generalizes substantially better.

## Repo layout

```
paper/      AAAI submission (main.tex + main.pdf, bib, AAAI style files)
code/       The MISD pipeline operators studied:
              misd_phase1_typing.py    structural typing (6-dim tags)
              misd_phase3_polish.py    rewrite rules R1–R5
              misd_phase4_safety.py    safety rules S1–S5 (rollback)
              misd_phase5_sympy.py     SymPy answer-equivalence
              misd_phase5_solve.py     solver harness
              misd_pipeline.py         end-to-end driver
results/    MISD_7Experiments.xlsx     full experiment workbook (configs, glossary, all tables)
            MISD_AAAI_findings.md      running findings notes
```

Solver/expert calls default to a local vLLM endpoint (`EXPERT_URL=http://localhost:8000/v1`).

## Status

Headline results are on Qwen2.5-3B. Generality on a second base scale (7B/1.5B) and a second source corpus are in progress (see `paper/main.tex` §Limitations).
