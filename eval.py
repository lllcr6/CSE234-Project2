"""
eval.py -- CSE/DSC 234 Project 2 schema-linking evaluation script.

Computes set-based Precision, Recall, and F1 at two granularities for each
question, then macro-averages across the question set:

  Table-level    : compares the set of TABLE identifiers used.
  Column-level   : compares the set of TABLE.COLUMN identifiers used,
                   i.e., a column is counted correct only if attributed to the
                   right table.

The leaderboard score is:

  Leaderboard Score = 0.5 * Table Score + 0.5 * Column Score
      Table Score    = (Precision_T + Recall_T + F1_T) / 3
      Column Score   = (Precision_C + Recall_C + F1_C) / 3

All identifiers are compared CASE-INSENSITIVELY to be robust to minor
casing inconsistencies in model outputs. Identifiers in your output that are
NOT present in the target db_id's schema (hallucinations) count as FALSE
POSITIVES, lowering your precision. They are additionally tallied in a
diagnostic "Schema-invalid identifiers: tables=X, columns=Y" line for your
debugging.

Boundary cases (per the project statement)
------------------------------------------
For each question, P/R/F1 are computed via standard set-based metrics with
the following conventions:
  - gold non-empty, pred non-empty: standard P, R, F1.
  - gold non-empty, pred empty:    P = 0, R = 0, F1 = 0 (recall miss).
  - gold empty,     pred empty:    P = 1, R = 1, F1 = 1 (perfect score).
  - gold empty,     pred non-empty: P = 0, R = 1, F1 = 0 (false positives;
                                     recall is vacuously 1 since gold is
                                     empty, but precision goes to 0).
In practice, gold sets in this project's data are only ever empty at the
column level (for SQL like ``select count(*) from t`` where the table is
referenced but no specific columns are), never at the table level.

Other robustness notes
----------------------
  - Non-list ``cols`` values (e.g. a string or null where a list is expected)
    are tolerated silently: the table is credited but no columns are counted.
    Emit lists.
  - ``question_id`` types must match between predictions and input
    (both ints, as produced by JSON parsing). A string ``question_id``
    will fail to match and the prediction will be scored as missing for
    that question; a WARN line surfaces the mismatch.
  - Predictions with duplicate question_ids: last entry wins (WARN logged).
  - Columns listed under a hallucinated table also count toward the
    column-hallucination tally (each contributes one ``__hallu__`` pair),
    so the diagnostic count may slightly overstate "column-only"
    hallucinations when entire tables are made up.

Usage
-----
    python eval.py \
        --predictions predictions.json \
        --gold validation_gold_schema_links.json \
        --schemas_dir schemas/ \
        --questions_input validation_input.json \
        [--per_question_out per_question.csv]

The --questions_input file is used to look up the db_id for each question_id
so that the script can verify predicted identifiers against the right schema.

The --gold file should contain a list of:
    {"question_id": <int>, "schema_links": {"Tbl": ["Col1", "Col2"], ...}}

The --predictions file should contain a list of:
    {"question_id": <int>, "schema_links": {"Tbl": ["Col1", "Col2"], ...}}

Both files are matched on question_id; predictions missing for a question_id
are scored as empty (zero precision/recall) for that question.
"""
import argparse
import json
import os
from collections import defaultdict


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_schema_caseinsensitive(schemas_dir, db_id):
    """Load a Spider-format schema and return:
        lc_tables : {lowercase_table -> original_table}
        lc_cols   : {original_table -> {lowercase_col -> original_col}}
    """
    fname = db_id.replace(' ', '_').replace('/', '_') + '.json'
    path = os.path.join(schemas_dir, fname)
    with open(path) as f:
        s = json.load(f)
    table_names = s['table_names_original']
    column_names = s['column_names_original']
    lc_tables = {t.lower(): t for t in table_names}
    lc_cols = defaultdict(dict)
    for tidx, cname in column_names:
        if tidx == -1:
            continue
        t = table_names[tidx]
        lc_cols[t][cname.lower()] = cname
    return lc_tables, dict(lc_cols)


def canonicalize_links(links, lc_tables, lc_cols, is_gold=False):
    """Convert a {table: [cols]} dict to (table_set, table_col_pair_set) of
    identifiers for set-based metric computation.

    For predictions (is_gold=False), schema-invalid identifiers (tables not in
    the schema, or columns not in a valid table's schema) are kept in the
    prediction set as un-matchable markers prefixed with ``__hallu__:``. They
    cannot match any gold identifier, so they correctly count as FALSE
    POSITIVES (lowering precision) while ALSO being reported as hallucinations
    for the student's debugging.

    For gold (is_gold=True), schema-invalid identifiers should never appear
    (ground truth is built from the same schema) but if they do, they are
    dropped silently rather than poisoning the gold set.
    """
    if not isinstance(links, dict):
        return set(), set(), {'bad_tables': 0, 'bad_columns': 0}
    tables = set()
    pairs = set()
    bad_tables = 0
    bad_columns = 0
    for t, cols in links.items():
        tlc = str(t).lower()
        if tlc in lc_tables:
            canonical_t = lc_tables[tlc]
            tables.add(canonical_t)
            cols_map = lc_cols.get(canonical_t, {})
            if isinstance(cols, list):
                for c in cols:
                    clc = str(c).lower()
                    if clc in cols_map:
                        pairs.add((canonical_t, cols_map[clc]))
                    else:
                        bad_columns += 1
                        if not is_gold:
                            # Add an un-matchable marker so it counts as FP
                            pairs.add((canonical_t, '__hallu__:' + clc))
        else:
            bad_tables += 1
            if not is_gold:
                # Hallucinated table -> marker, plus markers for any "columns"
                hallu_t = '__hallu__:' + tlc
                tables.add(hallu_t)
                if isinstance(cols, list):
                    for c in cols:
                        bad_columns += 1
                        pairs.add((hallu_t, '__hallu__:' + str(c).lower()))
    return tables, pairs, {
        'bad_tables': bad_tables,
        'bad_columns': bad_columns,
    }


def prf(pred_set, gold_set):
    """Standard set-based precision / recall / F1 with the following
    well-defined boundary cases:

      - tp + fp == 0  (pred empty): precision = 1 if gold is also empty,
                                    else 0.
      - tp + fn == 0  (gold empty): recall = 1 (nothing to find).
      - p + r == 0:                 F1 = 0.

    Concretely:
      pred=∅, gold=∅       → (1, 1, 1)   perfect on empty
      pred=∅, gold non-∅   → (0, 0, 0)   recall miss
      pred non-∅, gold=∅   → (0, 1, 0)   false positives; vacuous recall
      otherwise            → standard P, R, F1
    """
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if not gold_set else 0.0)
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predictions', required=True, help='Path to predictions JSON')
    ap.add_argument('--gold',        required=True, help='Path to gold schema_links JSON')
    ap.add_argument('--schemas_dir', required=True, help='Directory of Spider-format schema files')
    ap.add_argument('--questions_input', required=True, help='Input questions JSON (provides db_id per question_id)')
    ap.add_argument('--per_question_out', default=None, help='Optional CSV of per-question metrics')
    args = ap.parse_args()

    raw_preds = load_json(args.predictions)
    if not isinstance(raw_preds, list):
        print(f"ERROR: predictions file must be a JSON list, got {type(raw_preds).__name__}")
        return
    seen_qids = set()
    dup_qids = []
    preds = {}
    for p in raw_preds:
        qid = p.get('question_id')
        if qid in seen_qids:
            dup_qids.append(qid)
        seen_qids.add(qid)
        preds[qid] = p.get('schema_links', {})
    if dup_qids:
        print(f"WARN  predictions contain {len(dup_qids)} duplicate question_id(s) (last one wins): {dup_qids[:10]}")

    gold  = {g['question_id']: g['schema_links']         for g in load_json(args.gold)}
    questions = {q['question_id']: q for q in load_json(args.questions_input)}

    extra_qids = set(preds) - set(questions)
    if extra_qids:
        print(f"WARN  predictions contain {len(extra_qids)} question_id(s) not in input set (ignored): {sorted(extra_qids)[:10]}")

    rows = []
    sums = {k: 0.0 for k in ['p_t','r_t','f1_t','p_c','r_c','f1_c']}
    n_eval = 0
    hallucinations = {'bad_tables': 0, 'bad_columns': 0}
    missing_predictions = 0

    for qid, qmeta in sorted(questions.items()):
        db_id = qmeta['db_id']
        try:
            lc_tables, lc_cols = load_schema_caseinsensitive(args.schemas_dir, db_id)
        except FileNotFoundError:
            print(f"WARN  q{qid}: schema file not found for db_id {db_id!r}; skipping")
            continue
        if qid not in gold:
            print(f"WARN  q{qid}: not present in gold file; skipping")
            continue
        if qid not in preds:
            missing_predictions += 1
            pred_tables, pred_pairs, _bad = set(), set(), {'bad_tables': 0, 'bad_columns': 0}
        else:
            pred_tables, pred_pairs, bad = canonicalize_links(preds[qid], lc_tables, lc_cols, is_gold=False)
            hallucinations['bad_tables']  += bad['bad_tables']
            hallucinations['bad_columns'] += bad['bad_columns']

        gold_tables, gold_pairs, _ = canonicalize_links(gold[qid], lc_tables, lc_cols, is_gold=True)
        p_t, r_t, f1_t = prf(pred_tables, gold_tables)
        p_c, r_c, f1_c = prf(pred_pairs,  gold_pairs)
        rows.append({
            'question_id': qid, 'db_id': db_id,
            'p_t': p_t, 'r_t': r_t, 'f1_t': f1_t,
            'p_c': p_c, 'r_c': r_c, 'f1_c': f1_c,
        })
        sums['p_t']  += p_t;  sums['r_t']  += r_t;  sums['f1_t'] += f1_t
        sums['p_c']  += p_c;  sums['r_c']  += r_c;  sums['f1_c'] += f1_c
        n_eval += 1

    if n_eval == 0:
        print("No questions evaluated.")
        return

    avg = {k: v / n_eval for k, v in sums.items()}
    table_score  = (avg['p_t'] + avg['r_t'] + avg['f1_t']) / 3.0
    column_score = (avg['p_c'] + avg['r_c'] + avg['f1_c']) / 3.0
    leaderboard  = 0.5 * table_score + 0.5 * column_score

    print(f"Evaluated:                  {n_eval} questions")
    if missing_predictions:
        print(f"Missing prediction count:   {missing_predictions}  (scored as empty)")
    print(f"Schema-invalid identifiers: tables={hallucinations['bad_tables']}, columns={hallucinations['bad_columns']}")
    print("")
    print("---- Table-level (macro-averaged across questions) ----")
    print(f"  Precision_T : {avg['p_t']:.4f}")
    print(f"  Recall_T    : {avg['r_t']:.4f}")
    print(f"  F1_T        : {avg['f1_t']:.4f}")
    print(f"  Table Score : {table_score:.4f}     ((P+R+F1)/3)")
    print("")
    print("---- Column-level (Table.Column pairs, macro-averaged) ----")
    print(f"  Precision_C : {avg['p_c']:.4f}")
    print(f"  Recall_C    : {avg['r_c']:.4f}")
    print(f"  F1_C        : {avg['f1_c']:.4f}")
    print(f"  Column Score: {column_score:.4f}     ((P+R+F1)/3)")
    print("")
    print(f"==> Leaderboard Score : {leaderboard:.4f}   (0.5*Table + 0.5*Column)")

    if args.per_question_out:
        import csv
        with open(args.per_question_out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nPer-question metrics written to {args.per_question_out}")


if __name__ == '__main__':
    main()
