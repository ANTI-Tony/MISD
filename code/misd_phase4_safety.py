"""
MISD Phase 4: Math-Specific Safety Gate
5 formal rules to validate optimized instruction vs raw instruction.
Any violation -> revert to raw.

Usage:
    from misd_phase4_safety import safety_check
    verdict = safety_check(raw_ins, opt_ins, type_info)
    # verdict is 'kept' or 'revert_<reason>'

All rules are pure Python (regex/AST), no LLM dependency, fully reproducible.
"""
import re


LATEX_CMD_RE = re.compile(r'\\([a-zA-Z]+)')
NUMBER_RE = re.compile(r'-?\d+\.?\d*')
SINGLE_VAR_RE = re.compile(r'(?<![a-zA-Z\\])[a-zA-Z](?![a-zA-Z])')
MCQ_RE = re.compile(r'\(([A-E])\)')
ASY_BLOCK_RE = re.compile(r'\[asy\].*?\[/asy\]', re.DOTALL | re.IGNORECASE)


def _strip_asy(text):
    """Remove asy blocks before extracting features (asy contains its own vars/numbers)."""
    return ASY_BLOCK_RE.sub('', text)


def _normalize_latex_shorthand(text):
    """Expand \\frac12 -> \\frac{1}{2}, \\sqrt2 -> \\sqrt{2} so both shorthand and braced
    forms produce the same digit multiset."""
    text = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', text)
    text = re.sub(r'\\sqrt(\d)', r'\\sqrt{\1}', text)
    return text


def s1_latex_integrity(raw, opt):
    """LaTeX cmds in raw must all appear in opt (additions allowed); braces balanced."""
    raw_cmds = LATEX_CMD_RE.findall(_strip_asy(raw))
    opt_cmds = LATEX_CMD_RE.findall(_strip_asy(opt))
    opt_remaining = list(opt_cmds)
    for c in raw_cmds:
        if c in opt_remaining:
            opt_remaining.remove(c)
        else:
            return False, f's1_latex_cmd_dropped: \\{c}'
    if opt.count('{') != opt.count('}'):
        return False, 's1_latex_brace_unbalanced'
    return True, None


def _strip_mcq_and_asy(text):
    """Strip MCQ (A)~(E) markers and asy blocks before variable extraction."""
    return MCQ_RE.sub('', _strip_asy(text))


def s2_variable_consistency(raw, opt):
    """Raw single-letter vars must all appear in opt (post asy+MCQ strip; additions allowed)."""
    raw_vars = set(SINGLE_VAR_RE.findall(_strip_mcq_and_asy(raw)))
    opt_vars = set(SINGLE_VAR_RE.findall(_strip_mcq_and_asy(opt)))
    missing = raw_vars - opt_vars
    if missing:
        return False, f's2_vars_dropped: {missing}'
    return True, None


def s3_numerical_consistency(raw, opt):
    """All numeric literals in raw must appear in opt (after LaTeX shorthand normalization)."""
    raw_nums = sorted(NUMBER_RE.findall(_normalize_latex_shorthand(_strip_asy(raw))))
    opt_nums_list = NUMBER_RE.findall(_normalize_latex_shorthand(_strip_asy(opt)))
    # Check raw multiset is subset of opt multiset
    opt_remaining = list(opt_nums_list)
    for n in raw_nums:
        if n in opt_remaining:
            opt_remaining.remove(n)
        else:
            return False, f's3_number_dropped: {n}'
    return True, None


FORMAT_PHRASES = [
    '保留两位小数', '保留一位小数', '保留三位小数',
    '用分数', '以分数', '分数形式',
    '区间', '集合', '列表',
    'two decimal', 'three decimal', 'one decimal',
    'as a fraction', 'fraction form', 'simplest form',
    'common fraction', 'mixed number',
    '\\boxed', 'boxed',
    'integer', 'positive integer',
]


def s4_format_constraint(raw, opt, format_constraint=None):
    """Format constraint phrase (from Phase 1 typing) must persist if present in raw."""
    raw_lower = raw.lower()
    opt_lower = opt.lower()
    # Use Phase 1 hint if available
    if format_constraint:
        if format_constraint.lower() in raw_lower and format_constraint.lower() not in opt_lower:
            return False, f's4_format_lost: {format_constraint}'
        return True, None
    # Heuristic: check known format phrases
    for phrase in FORMAT_PHRASES:
        p = phrase.lower()
        if p in raw_lower and p not in opt_lower:
            return False, f's4_format_phrase_lost: {phrase}'
    return True, None


def s5_mcq_completeness(raw, opt):
    """All (A)..(E) options in raw must appear in opt."""
    raw_opts = set(MCQ_RE.findall(raw))
    opt_opts = set(MCQ_RE.findall(opt))
    if raw_opts and not raw_opts.issubset(opt_opts):
        missing = raw_opts - opt_opts
        return False, f's5_mcq_dropped: {missing}'
    return True, None


def safety_check(raw, opt, type_info=None):
    """Run all 5 rules. Return ('kept', None) or ('revert_<reason>', detail)."""
    if not opt or not opt.strip():
        return 'revert_empty', 'opt_empty'

    fmt = None
    if type_info and type_info.get('format', {}).get('present'):
        fmt = type_info['format'].get('constraint')

    checks = [
        ('s5', s5_mcq_completeness, (raw, opt)),
        ('s1', s1_latex_integrity, (raw, opt)),
        ('s2', s2_variable_consistency, (raw, opt)),
        ('s3', s3_numerical_consistency, (raw, opt)),
        ('s4', s4_format_constraint, (raw, opt, fmt)),
    ]
    for name, fn, args in checks:
        ok, detail = fn(*args)
        if not ok:
            return f'revert_{name}', detail
    return 'kept', None


# --- Self test ---
if __name__ == '__main__':
    cases = [
        # (raw, opt, expected_verdict_prefix)
        ("Solve $\\frac{x}{2} = 5$.",
         "Solve $\\frac{x}{2} = 5$ and provide \\boxed{} answer.",
         'kept'),
        ("Solve $\\frac{x}{2} = 5$.",
         "Solve x/2 = 5.",  # \frac dropped
         'revert_s1'),
        ("Variables a, b, c. Find a+b+c.",
         "Variables a, b. Find a+b.",  # c dropped
         'revert_s2'),
        ("If 5 + 3 = 8.",
         "If 5 + 4 = 9.",  # numbers changed
         'revert_s3'),
        ("Round to two decimal places: 1/3.",
         "Compute: 1/3.",  # format lost
         'revert_s4'),
        ("Choose: (A) 1 (B) 2 (C) 3.",
         "Choose: (A) 1 (B) 2.",  # (C) dropped
         'revert_s5'),
    ]
    print('=== MISD Phase 4 self-test ===')
    for i, (raw, opt, expect) in enumerate(cases):
        v, d = safety_check(raw, opt)
        ok = v.startswith(expect)
        print(f'Case {i}: {"PASS" if ok else "FAIL"}  verdict={v}  detail={d}')
        print(f'   raw: {raw}')
        print(f'   opt: {opt}')
