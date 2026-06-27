"""
MISD Phase 5: Sympy Answer-Equivalence Verifier
Run a solver on (raw, opt) pair, extract \\boxed{}, verify symbolic equivalence.

Usage offline (string-equivalence test only, no LLM call):
    from misd_phase5_sympy import answers_equivalent
    eq, reason = answers_equivalent("\\frac{1}{2}", "0.5")  # True
    eq, reason = answers_equivalent("3", "3.0")              # True

Server-side full pipeline (with LLM):
    python misd_phase5_sympy.py --pairs pairs.json --output verified.json
    where pairs.json = [{"id": ..., "raw": ..., "opt": ...}, ...]
"""
import re
import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr


BOXED_RE = re.compile(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}')


def extract_boxed(text):
    """Extract last \\boxed{...} content from text."""
    if not text:
        return None
    matches = BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_str(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r'\\text\{[^}]*\}', '', s)
    s = re.sub(r'\\left|\\right', '', s)
    s = re.sub(r'\\,|\\;|\\!|\\ ', '', s)
    s = s.replace('\\{', '{').replace('\\}', '}')
    s = s.replace(' ', '')
    s = s.replace('$', '')
    s = s.rstrip('.')
    return s


def _try_parse(s):
    """Try multiple parsing strategies, return sympy Expr or None."""
    if not s:
        return None
    # Strategy 1: parse_latex
    try:
        return parse_latex(s)
    except Exception:
        pass
    # Strategy 2: clean to plain math expr
    cleaned = s
    cleaned = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', cleaned)
    cleaned = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', cleaned)
    cleaned = re.sub(r'\\pi\b', 'pi', cleaned)
    cleaned = re.sub(r'\\cdot|\\times', '*', cleaned)
    cleaned = re.sub(r'\^', '**', cleaned)
    cleaned = re.sub(r'\\[a-zA-Z]+', '', cleaned)  # strip remaining commands
    cleaned = re.sub(r'[{}]', '', cleaned)
    try:
        return parse_expr(cleaned, evaluate=True)
    except Exception:
        return None


def answers_equivalent(a, b, atol=1e-9):
    """
    Decide if two boxed-answer strings are mathematically equivalent.
    Returns (bool, reason_str).
    """
    if a is None or b is None:
        return (a == b), 'both_none' if a is None and b is None else 'one_none'

    na, nb = _normalize_str(a), _normalize_str(b)
    if na == nb:
        return True, 'string_equal'

    # Try sympy parse + simplify
    ea, eb = _try_parse(a), _try_parse(b)
    if ea is not None and eb is not None:
        try:
            diff = sympy.simplify(ea - eb)
            if diff == 0:
                return True, 'sympy_equal'
            # Numeric fallback for transcendental cases
            try:
                d = float(diff.evalf())
                if abs(d) < atol:
                    return True, f'sympy_numeric_close: {d}'
            except Exception:
                pass
            return False, f'sympy_diff: {diff}'
        except Exception as e:
            pass

    # Set / interval / list literal compare (e.g., "{1, 2, 3}", "(1, \infty)")
    if na.startswith('{') and nb.startswith('{'):
        try:
            sa = set(re.findall(r'-?\d+\.?\d*', na))
            sb = set(re.findall(r'-?\d+\.?\d*', nb))
            if sa == sb and sa:
                return True, 'set_equal'
        except Exception:
            pass

    return False, 'no_match'


def verify_pair(raw_solution, opt_solution):
    """Given two model output texts, extract boxed and compare."""
    a_raw = extract_boxed(raw_solution)
    a_opt = extract_boxed(opt_solution)
    if a_raw is None and a_opt is None:
        return None, 'both_no_boxed'  # cannot verify; caller decides
    if a_raw is None or a_opt is None:
        return False, 'one_missing_boxed'
    eq, reason = answers_equivalent(a_raw, a_opt)
    return eq, f'{reason} | raw={a_raw!r} opt={a_opt!r}'


# --- Self test ---
if __name__ == '__main__':
    cases = [
        ("\\frac{1}{2}", "0.5", True),
        ("3", "3.0", True),
        ("\\frac{1}{2}", "\\frac{2}{4}", True),
        ("\\sqrt{4}", "2", True),
        ("\\pi", "pi", True),
        ("x+y", "y+x", True),
        ("1", "2", False),
        ("\\frac{1}{3}", "0.33", False),
        ("\\{1,2,3\\}", "\\{3,2,1\\}", True),
    ]
    print('=== MISD Phase 5 sympy verifier self-test ===')
    pass_n = 0
    for i, (a, b, expect) in enumerate(cases):
        eq, reason = answers_equivalent(a, b)
        ok = (eq == expect)
        pass_n += int(ok)
        print(f'Case {i}: {"PASS" if ok else "FAIL"}  {a!r} vs {b!r} -> {eq} ({reason})')
    print(f'\nTotal: {pass_n}/{len(cases)}')

    print('\n=== Boxed extraction ===')
    txts = [
        "Therefore the answer is \\boxed{42}.",
        "First \\boxed{5}, but actually \\boxed{6}.",
        "No box here.",
    ]
    for t in txts:
        print(f'  {t!r} -> {extract_boxed(t)!r}')
