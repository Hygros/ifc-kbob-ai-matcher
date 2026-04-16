from pathlib import Path
from collections import Counter
import random, json, re

queries_path = Path('Evaluation/ground_truth/queries.txt')
queries_text = queries_path.read_text(encoding='utf-8')
queries_lines = [ln.rstrip('\n') for ln in queries_text.splitlines()]

# ------------------------------------------------------------
# Heuristics for semantic slots requested by the user
# ------------------------------------------------------------
placements = {'insitu', 'precast', 'ortbeton', 'fertigteil'}
code_tokens = {'NPK'}
position_words = {'unten', 'oben', 'mitte', 'aussen', 'aussen.', 'unterer', 'oberer', 'lage'}
single_letter_codes = {'G', 'D', 'F', 'H', 'L'}

re_strength = re.compile(r'^(?:C\d+/\d+|S\d+[A-Z0-9]*|B\d+[A-Z0-9]*)$')
re_number = re.compile(r'^\d+(?:[.,]\d+)?$')
re_ordinal = re.compile(r'^\d+\.$')


def is_strength(tok: str) -> bool:
    return bool(re_strength.match(tok))


def is_placement(tok: str) -> bool:
    return tok.lower() in placements


def is_code(tok: str) -> bool:
    return tok in code_tokens or tok in single_letter_codes


def is_dimension(tok: str) -> bool:
    return bool(re_number.match(tok)) or bool(re_ordinal.match(tok))


def looks_like_predefined_type(tok: str) -> bool:
    if is_strength(tok) or is_placement(tok) or is_code(tok) or is_dimension(tok):
        return False
    if tok.startswith('Ifc'):
        return False
    return any(ch.isalpha() for ch in tok)


def classify_line(line: str):
    toks = line.split()
    if not toks:
        return {'tokens': toks, 'predefined': [], 'material': [], 'strength': [], 'placement': [], 'all_eligible': []}

    predefined = []
    material = []
    strength = []
    placement = []

    if len(toks) >= 2 and looks_like_predefined_type(toks[1]):
        predefined.append(1)

    for i, tok in enumerate(toks[1:], start=1):
        if i == 1 and i in predefined:
            continue
        if is_strength(tok):
            strength.append(i)
        elif is_placement(tok):
            placement.append(i)
        else:
            if is_code(tok) or is_dimension(tok):
                continue
            if tok.lower() in position_words:
                continue
            if tok.startswith('Ifc'):
                continue
            if any(ch.isalpha() for ch in tok):
                material.append(i)

    material = [i for i in material if i not in predefined]
    eligible = predefined + material + strength + placement
    return {
        'tokens': toks,
        'predefined': sorted(predefined),
        'material': sorted(material),
        'strength': sorted(strength),
        'placement': sorted(placement),
        'all_eligible': sorted(set(eligible)),
    }

# ------------------------------------------------------------
# Human-like typo operations, now max 1 typo per word/token
# ------------------------------------------------------------
neighbors = {
    'a': list('qwsyz'), 'b': list('vghn'), 'c': list('xdfv'), 'd': list('serfcx'),
    'e': list('wsdr34'), 'f': list('drtgcv'), 'g': list('ftyhbv'), 'h': list('gzujnb'),
    'i': list('uojk89'), 'j': list('huikmn'), 'k': list('jiolm,'), 'l': list('kopö.-'),
    'm': list('njk,'), 'n': list('bhjm'), 'o': list('iklp90'), 'p': list('olüß0-'),
    'q': list('wa12'), 'r': list('edft45'), 's': list('awedxy'), 't': list('rfgz56'),
    'u': list('zhji78'), 'v': list('cfgb'), 'w': list('qase23'), 'x': list('ysdc'),
    'y': list('xsdaz'), 'z': list('tuah'), 'ä': list('öl'), 'ö': list('üäp'), 'ü': list('öp')
}


def is_letter(c: str) -> bool:
    return c.isalpha() or c in 'äöüÄÖÜ'


def eligible_positions(tok: str):
    return [i for i, ch in enumerate(tok) if ch.isalnum() or ch in 'äöüÄÖÜ']


def op_delete(tok, rng):
    pos = eligible_positions(tok)
    if len(pos) < 2:
        return None
    i = rng.choice(pos)
    return tok[:i] + tok[i+1:]


def op_transpose(tok, rng):
    candidates = [i for i in range(len(tok)-1)
                  if (tok[i].isalnum() or is_letter(tok[i])) and (tok[i+1].isalnum() or is_letter(tok[i+1]))]
    if not candidates:
        return None
    i = rng.choice(candidates)
    return tok[:i] + tok[i+1] + tok[i] + tok[i+2:]


def op_duplicate(tok, rng):
    pos = eligible_positions(tok)
    if not pos:
        return None
    i = rng.choice(pos)
    return tok[:i+1] + tok[i] + tok[i+1:]


def op_substitute(tok, rng):
    pos = eligible_positions(tok)
    if not pos:
        return None
    i = rng.choice(pos)
    ch = tok[i]
    lower = ch.lower()
    repls = neighbors.get(lower)
    if repls:
        rep = rng.choice(repls)
        if ch.isupper():
            rep = rep.upper()
        return tok[:i] + rep + tok[i+1:]
    if ch.isdigit():
        rep = rng.choice([d for d in '0123456789' if d != ch])
        return tok[:i] + rep + tok[i+1:]
    alpha = 'abcdefghijklmnopqrstuvwxyz'
    if lower in alpha:
        idx = alpha.index(lower)
        rep = alpha[max(0, min(len(alpha)-1, idx + rng.choice([-1, 1])))]
        if ch.isupper():
            rep = rep.upper()
        return tok[:i] + rep + tok[i+1:]
    return None


def op_insert(tok, rng):
    pos = eligible_positions(tok)
    if not pos:
        return None
    i = rng.choice(pos)
    base = tok[i]
    lower = base.lower()
    repls = neighbors.get(lower)
    if repls:
        ins = rng.choice(repls)
        if base.isupper():
            ins = ins.upper()
    elif base.isdigit():
        ins = rng.choice('0123456789')
    else:
        ins = base
    return tok[:i+1] + ins + tok[i+1:]

ops = [op_delete, op_transpose, op_duplicate, op_substitute, op_insert]
op_names = {
    op_delete: 'delete',
    op_transpose: 'transpose',
    op_duplicate: 'duplicate',
    op_substitute: 'substitute',
    op_insert: 'insert',
}


def token_weight(tok: str, category: str) -> int:
    base = sum(3 for c in tok if is_letter(c)) + sum(1 for c in tok if c.isdigit())
    boost = {'predefined': 3, 'material': 3, 'strength': 2, 'placement': 2}.get(category, 1)
    return max(1, base + boost)

# ------------------------------------------------------------
# 1) Missing one token per line, only allowed semantic slots
# ------------------------------------------------------------
seed_missing = 61
rng_missing = random.Random(seed_missing)
missing_lines = []
missing_records = []
missing_category_counter = Counter()
missing_impossible = 0

for ln in queries_lines:
    cls = classify_line(ln)
    toks = cls['tokens']
    if not toks:
        missing_lines.append('')
        missing_records.append({'original': ln, 'modified': '', 'removed_index': None, 'removed_token': None, 'removed_category': None})
        continue
    category_candidates = [cat for cat in ['predefined', 'material', 'strength', 'placement'] if cls[cat]]
    if not category_candidates:
        missing_impossible += 1
        missing_lines.append(ln)
        missing_records.append({'original': ln, 'modified': ln, 'removed_index': None, 'removed_token': None, 'removed_category': None})
        continue
    chosen_cat = rng_missing.choice(category_candidates)
    remove_idx = rng_missing.choice(cls[chosen_cat])
    removed_tok = toks[remove_idx]
    new_toks = toks[:remove_idx] + toks[remove_idx+1:]
    new_ln = ' '.join(new_toks)
    missing_lines.append(new_ln)
    missing_records.append({'original': ln, 'modified': new_ln, 'removed_index': remove_idx, 'removed_token': removed_tok, 'removed_category': chosen_cat})
    missing_category_counter[chosen_cat] += 1

missing_path = Path('queries_missing_one_per_line.txt')
missing_path.write_text('\n'.join(missing_lines) + '\n', encoding='utf-8')

# ------------------------------------------------------------
# 2) 1-2 typos per line, but max 1 typo per word/token
# ------------------------------------------------------------
seed_typos = 72
rng_typos = random.Random(seed_typos)
typos_lines = []
typos_records = []
typos_count_counter = Counter()
typos_target_category_counter = Counter()
typos_op_counter = Counter()
typos_impossible = 0

for ln in queries_lines:
    cls = classify_line(ln)
    toks = cls['tokens']
    if not toks:
        typos_lines.append('')
        typos_records.append({'original': ln, 'modified': '', 'typo_count': 0, 'changes': []})
        continue

    eligible = []
    for cat in ['predefined', 'material', 'strength', 'placement']:
        for idx in cls[cat]:
            eligible.append((idx, cat))

    if not eligible:
        typos_impossible += 1
        typos_lines.append(ln)
        typos_records.append({'original': ln, 'modified': ln, 'typo_count': 0, 'changes': []})
        continue

    new_toks = toks[:]
    typo_target = rng_typos.randint(1, min(2, len(eligible)))
    available = eligible[:]  # each token can be picked at most once => max 1 typo per word
    changes = []
    for _ in range(typo_target):
        if not available:
            break
        weights = [token_weight(new_toks[idx], cat) for idx, cat in available]
        pick_idx = rng_typos.choices(range(len(available)), weights=weights, k=1)[0]
        token_idx, cat = available.pop(pick_idx)
        old_tok = new_toks[token_idx]
        op_order = ops[:]
        rng_typos.shuffle(op_order)
        new_tok = None
        used_op = None
        for op in op_order:
            cand = op(old_tok, rng_typos)
            if cand is not None and cand != old_tok:
                new_tok = cand
                used_op = op_names[op]
                break
        if new_tok is None:
            pos = eligible_positions(old_tok)
            if pos:
                i = pos[-1]
                new_tok = old_tok[:i+1] + old_tok[i] + old_tok[i+1:]
                used_op = 'duplicate_fallback'
            else:
                continue
        new_toks[token_idx] = new_tok
        changes.append({'token_index': token_idx, 'category': cat, 'original_token': old_tok, 'modified_token': new_tok, 'operation': used_op})
        typos_target_category_counter[cat] += 1
        typos_op_counter[used_op] += 1

    new_ln = ' '.join(new_toks)
    typos_lines.append(new_ln)
    typos_records.append({'original': ln, 'modified': new_ln, 'typo_count': len(changes), 'changes': changes})
    if changes:
        typos_count_counter[len(changes)] += 1

# validate max 1 typo per word
for rec in typos_records:
    idxs = [c['token_index'] for c in rec['changes']]
    assert len(idxs) == len(set(idxs))

typos_path = Path('queries_typos_1to2_per_line.txt')
typos_path.write_text('\n'.join(typos_lines) + '\n', encoding='utf-8')

# ------------------------------------------------------------
# 3) Combined: allowed missing token + allowed typos, max 1 typo per word
# ------------------------------------------------------------
seed_combined = 73
rng_combined = random.Random(seed_combined)
combined_lines = []
combined_records = []
combined_count_counter = Counter()
combined_target_category_counter = Counter()
combined_op_counter = Counter()
combined_impossible_typos = 0

for rec in missing_records:
    original = rec['original']
    after_removal = rec['modified']
    cls = classify_line(after_removal)
    toks = cls['tokens']

    eligible = []
    for cat in ['predefined', 'material', 'strength', 'placement']:
        for idx in cls[cat]:
            eligible.append((idx, cat))

    if not toks or not eligible:
        if original.split() and not eligible:
            combined_impossible_typos += 1
        combined_lines.append(after_removal)
        combined_records.append({
            'original': original,
            'after_removal': after_removal,
            'modified': after_removal,
            'removed_index': rec['removed_index'],
            'removed_token': rec['removed_token'],
            'removed_category': rec['removed_category'],
            'typo_count': 0,
            'changes': [],
        })
        continue

    new_toks = toks[:]
    typo_target = rng_combined.randint(1, min(2, len(eligible)))
    available = eligible[:]  # each token can be picked at most once => max 1 typo per word
    changes = []
    for _ in range(typo_target):
        if not available:
            break
        weights = [token_weight(new_toks[idx], cat) for idx, cat in available]
        pick_idx = rng_combined.choices(range(len(available)), weights=weights, k=1)[0]
        token_idx, cat = available.pop(pick_idx)
        old_tok = new_toks[token_idx]
        op_order = ops[:]
        rng_combined.shuffle(op_order)
        new_tok = None
        used_op = None
        for op in op_order:
            cand = op(old_tok, rng_combined)
            if cand is not None and cand != old_tok:
                new_tok = cand
                used_op = op_names[op]
                break
        if new_tok is None:
            pos = eligible_positions(old_tok)
            if pos:
                i = pos[-1]
                new_tok = old_tok[:i+1] + old_tok[i] + old_tok[i+1:]
                used_op = 'duplicate_fallback'
            else:
                continue
        new_toks[token_idx] = new_tok
        changes.append({'token_index_after_removal': token_idx, 'category': cat, 'original_token': old_tok, 'modified_token': new_tok, 'operation': used_op})
        combined_target_category_counter[cat] += 1
        combined_op_counter[used_op] += 1

    modified = ' '.join(new_toks)
    combined_lines.append(modified)
    combined_records.append({
        'original': original,
        'after_removal': after_removal,
        'modified': modified,
        'removed_index': rec['removed_index'],
        'removed_token': rec['removed_token'],
        'removed_category': rec['removed_category'],
        'typo_count': len(changes),
        'changes': changes,
    })
    if changes:
        combined_count_counter[len(changes)] += 1

# validate max 1 typo per word
for rec in combined_records:
    idxs = [c['token_index_after_removal'] for c in rec['changes']]
    assert len(idxs) == len(set(idxs))

combined_path = Path('queries_missing_one_and_typos_1to2_per_line.txt')
combined_path.write_text('\n'.join(combined_lines) + '\n', encoding='utf-8')

# ------------------------------------------------------------
# Validation and summary
# ------------------------------------------------------------
analysis_counter = Counter()
lines_with_any_allowed_slot = 0
for ln in queries_lines:
    cls = classify_line(ln)
    for cat in ['predefined', 'material', 'strength', 'placement']:
        if cls[cat]:
            analysis_counter[cat] += 1
    if cls['all_eligible']:
        lines_with_any_allowed_slot += 1

nonempty_queries = sum(1 for ln in queries_lines if ln.split())
changed_missing = sum(1 for a, b in zip(queries_lines, missing_lines) if a != b)
changed_typos = sum(1 for a, b in zip(queries_lines, typos_lines) if a != b)
changed_combined = sum(1 for a, b in zip(queries_lines, combined_lines) if a != b)

assert changed_missing == nonempty_queries - missing_impossible
assert changed_typos == nonempty_queries - typos_impossible
for rec in typos_records:
    if rec['changes']:
        assert 1 <= rec['typo_count'] <= 2
for rec in combined_records:
    if rec['changes']:
        assert 1 <= rec['typo_count'] <= 2

summary = {
    'queries_file': str(queries_path),
    'queries_line_count': len(queries_lines),
    'nonempty_queries': nonempty_queries,
    'rule_update': 'Typos are additionally limited to at most one typo per word/token. IfcEntity remains untouched.',
    'classification_rule': {
        'entity_preserved': 'Token 0 (IfcEntity) is never removed or typo-modified.',
        'predefinedtype': 'Primarily token 1, if it is not a code, number, strength class or placement token.',
        'material': 'Alphabetic/compound payload tokens excluding entity, predefined type, NPK codes, pure numbers and common positional words.',
        'strength': 'Tokens like C30/37, S235JR, B500B.',
        'placement': 'insitu, precast, Ortbeton, Fertigteil.',
    },
    'lines_with_allowed_slots': lines_with_any_allowed_slot,
    'lines_by_slot_presence': dict(sorted(analysis_counter.items())),
    'seeds': {
        'missing': seed_missing,
        'typos': seed_typos,
        'combined': seed_combined,
    },
    'outputs': {
        'missing_one_per_line': str(missing_path),
        'typos_1to2_per_line': str(typos_path),
        'missing_one_and_typos_1to2_per_line': str(combined_path),
    },
    'missing': {
        'changed_lines': changed_missing,
        'unchanged_lines_due_to_no_allowed_slot': missing_impossible,
        'removed_category_distribution': dict(sorted(missing_category_counter.items())),
    },
    'typos': {
        'changed_lines': changed_typos,
        'unchanged_lines_due_to_no_allowed_slot': typos_impossible,
        'typo_count_distribution': dict(sorted(typos_count_counter.items())),
        'target_category_distribution': dict(sorted(typos_target_category_counter.items())),
        'operation_distribution': dict(sorted(typos_op_counter.items())),
        'max_typos_per_word': 1,
    },
    'combined': {
        'changed_lines': changed_combined,
        'lines_without_typos_after_removal': combined_impossible_typos,
        'typo_count_distribution': dict(sorted(combined_count_counter.items())),
        'target_category_distribution': dict(sorted(combined_target_category_counter.items())),
        'operation_distribution': dict(sorted(combined_op_counter.items())),
        'max_typos_per_word': 1,
    },
    'examples_missing': missing_records[:12],
    'examples_typos': typos_records[:12],
    'examples_combined': combined_records[:12],
}
summary_path = Path('queries_regeneration_summary_restricted_slots_max1typo_per_word.json')
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

print(json.dumps({
    'outputs': summary['outputs'],
    'missing_changed_lines': changed_missing,
    'typos_changed_lines': changed_typos,
    'combined_changed_lines': changed_combined,
    'typos_distribution': summary['typos']['typo_count_distribution'],
    'combined_distribution': summary['combined']['typo_count_distribution'],
    'max_typos_per_word': 1,
    'unchanged_due_to_no_allowed_slot': {
        'missing': missing_impossible,
        'typos': typos_impossible,
        'combined_after_removal': combined_impossible_typos,
    }
}, ensure_ascii=False))