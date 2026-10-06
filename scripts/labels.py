"""QualiSpeech fault-label parser (VLSP 2026 SQA).

Turns the free-text fault columns of QualiSpeech
(``Noise Description``, ``Distortion description``, ``Unnatural pause``)
into structured time intervals, one record per (file, fault).

Design rules (agreed in the team label pipeline):
  * The parser only converts text -> intervals. It never invents an interval
    from ambiguous text; such rows get status ``ambiguous`` and empty
    ``intervals`` (a best guess, if any, goes to ``candidates`` for review).
  * Every typo fix is an explicit, named rule, recorded in ``rules``.
  * Only rows with status in USABLE_STATUSES may be used for training/eval
    before human audit.

Standalone on purpose: only the standard library + pandas, no other ``sqa``
imports, so it runs in Colab with just this file and ``train.csv``.

Usage (Colab / shell):
    python labels.py train.csv --out label_out
which writes ``label_out/parsed_labels.jsonl`` (all records, for Kiên),
``label_out/review_queue.csv`` (rows to audit, for Hiếu) and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: fault name -> (text column, matching score column). Score 5 means "clean".
FAULT_COLUMNS = {
    "noise": ("Noise Description", "Background noise"),
    "distortion": ("Distortion description", "Distortion"),
    "unnatural_pause": ("Unnatural pause", "Continuity"),
}
DESCRIPTION_COLUMN = "Natural language description"
ID_COLUMN = "id"

#: statuses whose intervals can be used without human audit
USABLE_STATUSES = ("no_fault", "parsed", "normalized", "audited", "rule_confirmed")
#: statuses that always go to the review queue
REVIEW_STATUSES = ("ambiguous", "abnormal", "disputed")
#: fault is present but has no usable time: count it for presence (F1), skip it for localization
PRESENCE_ONLY_STATUSES = ("present_unknown_time",)

#: sanity bound for timestamps (longest QualiSpeech clips are ~19 s)
MAX_PLAUSIBLE_SECONDS = 30.0

#: ``None`` as an interval end means "end of clip"; it is resolved later,
#: once real audio durations are known.
END_OF_CLIP = None

# --------------------------------------------------------------------------
# Regular expressions
# --------------------------------------------------------------------------

NUM = r"(\d+(?:\.\d+)?)"
UNIT = r"(?:\s*(?:s|sec|secs|second|seconds)\b)?"
SEP = r"\s*(?:-|–|—|~|to|until)\s*"
RANGE_RE = re.compile(NUM + UNIT + SEP + NUM + UNIT, re.I)
NUM_RE = re.compile(NUM)

AFTER_RE = re.compile(r"\bafter\s+(?:the\s+)?" + NUM + UNIT, re.I)
WHOLE_RE = re.compile(r"\b(?:whole|entire|throughout|all the time)\b", re.I)
NO_SPEECH_RE = re.compile(r"can'?t hear", re.I)

#: "no fault" phrases, matched on letters-only text (typos included)
NONE_PATTERNS = [
    re.compile(r"^(?:not?|no)\s*not?ic\w*$"),     # not noticeable, notnoticeable, not noticebale
    re.compile(r"^(?:not?|no)\s*n[a-z]*(?:ble|bale|abel)$"),  # not notcieable, not nticeable
    re.compile(r"^(?:not?|no)\s*noise$"),          # no noise
    re.compile(r"^(?:not?|no)\s*dis\w*$"),         # not distorted, not distored, no distortion
    re.compile(r"^(?:\w*ery\s*|ve\w*y\s*|mostly\s*)?smooth\w*$"),  # very smooth, vey smooth, smoothy
    re.compile(r"^(?:none|no|nothing)$"),
]

#: typo fixes: (rule name, pattern, replacement). Order matters.
TYPO_RULES = [
    ("comma_decimal", re.compile(r"(\d),(\d)"), None),                   # 8,9 -> 8.9 (see _comma)
    ("double_dot", re.compile(r"(\d)\.\.(\d)"), r"\1.\2"),             # 2..5 -> 2.5
    ("space_before_decimal", re.compile(r"(\d)\s+\.(\d)"), r"\1.\2"),  # 3 .5 -> 3.5
    ("letter_o_as_zero", re.compile(r"\b(from|at)\s+o\b|^o(?=\s*(?:-|to)\b)", re.I),
     lambda m: (m.group(1) + " 0") if m.group(1) else "0"),            # from o to 1.6
    ("t_as_to", re.compile(r"(\d)\s*s?\s+t\s+(\d)", re.I), r"\1 to \2"),  # 0.3 t 2.7
    ("glued_to", re.compile(r"(\d)(?:s)?to(?=\s*\d)|(?<=to)(?=\d)", re.I),
     lambda m: (m.group(1) + " to") if m.group(1) else " "),          # 1to 6.9, to6
    ("missing_decimal", re.compile(r"(?<![\d.])0(\d)(?![\d.])"), r"0.\1"),  # 05-2.4s -> 0.5
    ("double_dash", re.compile(r"-{2,}"), "-"),
    ("missing_separator", re.compile(r"\bfrom\s+" + NUM + r"\s+" + NUM, re.I),
     r"from \1 to \2"),                                                # from 0 11.9 seconds
]

#: format variants that are not typos (status stays "parsed")
FORMAT_RULES = [
    ("between_and", re.compile(r"\bbetween\s+" + NUM + UNIT + r"\s+and\s+", re.I),
     r"between \1 to "),
    ("trailing_dot", re.compile(r"(\d)\.(?!\d)"), r"\1"),             # "0. to 2" / "6.9."
]

#: heuristics for text written in the wrong column (voice descriptions)
WRONG_COLUMN_RE = re.compile(r"\bvoice\b.*\b(?:tone|sounds?|man|woman|girl|boy)\b"
                             r"|\b(?:man|woman)'s\b.*\bvoice\b", re.I)

#: the pause column should only hold times; these words mean another fault's text
OTHER_FAULT_WORDING = {
    "unnatural_pause": re.compile(r"distort|mechanical|noise|electric|buzz|crackl", re.I),
}

#: keywords to pick the matching sentence from the natural language description
DESCRIPTION_KEYWORDS = {
    "noise": ("noise", "background", "music", "sound of"),
    "distortion": ("distort", "mechanical", "abrupt", "electric"),
    "unnatural_pause": ("pause", "smooth", "discontinu", "disjoint", "interrupt",
                        "break", "stutter", "flow"),
}


# --------------------------------------------------------------------------
# Core parsing
# --------------------------------------------------------------------------

def _comma(text: str) -> str:
    """Decimal comma -> dot, except when the comma separates two ranges,
    e.g. 'from 2 to 4,7 to 8 seconds' means 2-4 and 7-8, not 4.7."""
    def repl(m):
        before, after = text[:m.start()], text[m.end():]
        ends_a_range = re.search(r"(?:-|to)\s*\d+$", before + m.group(1))
        starts_a_range = re.match(r"\d*\s*(?:s\b)?\s*(?:-|~|to)\s*\d", m.group(2) + after)
        if ends_a_range and starts_a_range:
            return m.group(1) + ", " + m.group(2)
        return m.group(1) + "." + m.group(2)
    return re.sub(r"(\d),(\d)", repl, text)


def normalize_text(raw: str) -> tuple[str, list[str], list[str]]:
    """Lowercase and repair known typos.

    Returns (clean_text, typo_rules_fired, format_rules_fired).
    """
    text = " ".join(str(raw).strip().lower().split())
    typo_fired, format_fired = [], []
    for name, pattern, repl in TYPO_RULES:
        new = _comma(text) if repl is None else pattern.sub(repl, text)
        if new != text:
            typo_fired.append(name)
            text = new
    for name, pattern, repl in FORMAT_RULES:
        new = pattern.sub(repl, text)
        if new != text:
            format_fired.append(name)
            text = new
    return text, typo_fired, format_fired


def is_no_fault(text: str) -> bool:
    letters = re.sub(r"[^a-z\s]", " ", text.lower())
    letters = " ".join(letters.split())
    return any(p.match(letters) for p in NONE_PATTERNS)


def _merge(intervals: list[list[Optional[float]]]) -> list[list[Optional[float]]]:
    """Sort and merge overlapping/touching intervals. None end = end of clip."""
    def end_key(e):
        return float("inf") if e is None else e
    out: list[list[Optional[float]]] = []
    for s, e in sorted(intervals, key=lambda x: (x[0], end_key(x[1]))):
        if out and s <= end_key(out[-1][1]):
            if end_key(e) > end_key(out[-1][1]):
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def _record(raw, status, intervals=None, candidates=None, rules=None,
            reason="", hint=""):
    return {
        "status": status,
        "intervals": intervals or [],
        "candidates": candidates or [],
        "rules": rules or [],
        "reason": reason,
        "hint": hint,
        "raw": "" if raw is None or (isinstance(raw, float) and pd.isna(raw)) else str(raw),
    }


def parse_cell(raw, fault: str = "") -> dict:
    """Parse one fault cell into a record (without file id / score fields).

    Status meanings:
      no_fault    text says the fault is absent -> intervals []
      parsed      clean time range(s) found as written
      normalized  time range(s) found after a named typo fix (see ``rules``)
      ambiguous   fault mentioned but time unclear (point, no time, extra numbers)
      abnormal    unusable text (malformed range, wrong column, no speech, empty)
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or not str(raw).strip():
        return _record(raw, "abnormal", reason="empty")

    raw_s = str(raw)
    if is_no_fault(raw_s):
        return _record(raw, "no_fault")

    if NO_SPEECH_RE.search(raw_s):
        return _record(raw, "abnormal", reason="no_speech")

    text, typo_rules, format_rules = normalize_text(raw_s)
    rules = typo_rules + format_rules

    ranges = [[round(float(a), 3), round(float(b), 3)] for a, b in RANGE_RE.findall(text)]

    other = OTHER_FAULT_WORDING.get(fault)
    if other and other.search(text) and not is_no_fault(raw_s):
        return _record(raw, "abnormal", candidates=ranges, rules=rules,
                       reason="wrong_column: describes another fault")
    leftover = RANGE_RE.sub(" ", text)
    after = [[round(float(x), 3), END_OF_CLIP] for x in AFTER_RE.findall(leftover)]
    leftover = AFTER_RE.sub(" ", leftover)
    extra_numbers = NUM_RE.findall(leftover)

    # malformed ranges: start >= end, or implausible values
    bad = [r for r in ranges if r[0] >= r[1] or r[1] > MAX_PLAUSIBLE_SECONDS]
    if bad:
        return _record(raw, "abnormal", candidates=ranges, rules=rules,
                       reason="malformed_range: " + ", ".join(f"{a:g}-{b:g}" for a, b in bad))

    found = ranges + after
    if found and not extra_numbers:
        status = "normalized" if typo_rules else "parsed"
        if after:
            rules = rules + ["after_x_to_end"]
        return _record(raw, status, intervals=_merge(found), rules=rules)

    if found and extra_numbers:
        return _record(raw, "ambiguous", candidates=_merge(found), rules=rules,
                       reason="extra_number: " + ", ".join(extra_numbers))

    # no range at all from here on
    if extra_numbers:
        return _record(raw, "ambiguous", rules=rules,
                       reason="point_or_duration: " + ", ".join(extra_numbers))

    if WRONG_COLUMN_RE.search(text):
        return _record(raw, "abnormal", rules=rules, reason="wrong_column")

    if WHOLE_RE.search(text):
        return _record(raw, "normalized" if typo_rules else "parsed",
                       intervals=[[0.0, END_OF_CLIP]], rules=rules + ["whole_audio"])

    # free text with no number: fault mentioned without time, or unknown text
    if fault in ("noise", "distortion") and re.search(r"\bthere (?:is|are)\b", text):
        return _record(raw, "ambiguous", rules=rules, reason="present_no_time")
    return _record(raw, "abnormal", rules=rules, reason="unrecognized_text")


# --------------------------------------------------------------------------
# Cross-checks
# --------------------------------------------------------------------------

def check_conflict(status: str, score) -> tuple[bool, str]:
    """Compare text with its score (5 = clean). Returns (conflict, reason)."""
    try:
        score = int(score)
    except (TypeError, ValueError):
        return False, ""
    says_fault = status not in ("no_fault",) and status != "abnormal"
    if status == "no_fault" and score < 5:
        return True, f"text says no fault but score={score}"
    if says_fault and score == 5:
        return True, "text describes a fault but score=5"
    return False, ""


def _cross_column_hint(rec: dict, row: pd.Series) -> str:
    """Evidence from another column; never applied automatically."""
    if rec["fault"] != "unnatural_pause" or rec["status"] != "ambiguous":
        return ""
    m = re.fullmatch(r"\s*" + NUM + r"\s*[a-z]*\s*[.,]?\s*", rec["raw"].lower())
    if not m:
        return ""
    x = float(m.group(1))
    dist_raw = str(row.get(FAULT_COLUMNS["distortion"][0], "")).lower()
    after = [float(v) for v in AFTER_RE.findall(dist_raw)]
    if x in after:
        return f"distortion column says 'after {x:g}s' -> proposed [{x:g}, end_of_clip]"
    return ""


def describe_sentence(description, fault: str) -> str:
    """Sentences of the natural-language description that mention this fault."""
    if not isinstance(description, str):
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", description)
    keys = DESCRIPTION_KEYWORDS.get(fault, ())
    hits = [s for s in sentences if any(k in s.lower() for k in keys)]
    return " ".join(hits[:2])


# --------------------------------------------------------------------------
# DataFrame level
# --------------------------------------------------------------------------

def parse_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """One output row per (file, fault)."""
    out = []
    for _, row in df.iterrows():
        for fault, (text_col, score_col) in FAULT_COLUMNS.items():
            rec = parse_cell(row.get(text_col), fault)
            rec["file_id"] = row[ID_COLUMN]
            rec["fault"] = fault
            rec["score"] = row.get(score_col)
            rec["conflict"], rec["conflict_reason"] = check_conflict(rec["status"], rec["score"])
            if rec["status"] == "ambiguous":
                rec["hint"] = _cross_column_hint(rec, row)
            out.append(rec)
    cols = ["file_id", "fault", "status", "intervals", "candidates", "rules",
            "reason", "hint", "conflict", "conflict_reason", "score", "raw"]
    result = pd.DataFrame(out)[cols]
    result["label_source"] = "parser"
    result["audit_note"] = ""
    result["label_version"] = "v1"
    return result


def review_queue(records: pd.DataFrame, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rows Hiếu should audit: ambiguous, abnormal, or score conflicts."""
    mask = records["status"].isin(REVIEW_STATUSES) | records["conflict"]
    if "label_source" in records.columns:
        mask &= ~((records["label_source"] == "human_audit") & ~records["status"].isin(REVIEW_STATUSES))
    q = records[mask].copy()
    q["why"] = q.apply(
        lambda r: "; ".join(x for x in [r["status"] if r["status"] in REVIEW_STATUSES else "",
                                         r["reason"], r["conflict_reason"]] if x), axis=1)
    if df is not None and DESCRIPTION_COLUMN in df.columns:
        desc = df.set_index(ID_COLUMN)[DESCRIPTION_COLUMN]
        q["description_sentence"] = [describe_sentence(desc.get(fid), f)
                                     for fid, f in zip(q["file_id"], q["fault"])]
    else:
        q["description_sentence"] = ""
    q["candidates"] = q["candidates"].map(format_intervals)
    q["parsed_intervals"] = q["intervals"].map(format_intervals)
    # empty columns for the auditor
    q["verdict"] = ""               # accept / reject / correct
    q["corrected_intervals"] = ""   # e.g. 1.2-1.8;3.0-3.4  (use 'end' for end of clip)
    q["reviewer_note"] = ""
    cols = ["file_id", "fault", "status", "why", "raw", "parsed_intervals", "candidates",
            "hint", "score", "description_sentence", "verdict", "corrected_intervals",
            "reviewer_note"]
    return q[cols].sort_values(["fault", "status", "file_id"]).reset_index(drop=True)


def format_intervals(intervals) -> str:
    """[[0.5, 2.4], [3.0, None]] -> '0.5-2.4;3-end'."""
    return ";".join(f"{s:g}-{'end' if e is None else format(e, 'g')}" for s, e in intervals)


def summary(records: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(records["status"], records["fault"], margins=True, margins_name="total")
    order = [s for s in ("no_fault", "parsed", "normalized", "audited", "rule_confirmed",
                         "present_unknown_time", "ambiguous", "abnormal", "disputed", "total")
             if s in table.index]
    return table.loc[order]


# --------------------------------------------------------------------------
# Applying human audit verdicts (Hiếu's return file)
# --------------------------------------------------------------------------

AUDIT_DECISIONS = ("accept", "localized", "absent", "present_unknown_time", "unknown")


def parse_interval_text(text) -> list[list[Optional[float]]]:
    """'0.5-2.4;6.9-end' -> [[0.5, 2.4], [6.9, None]]. Ignores a 'candidate ' prefix."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    text = str(text).strip().lower().replace("candidate", "").strip()
    if text in ("", "—", "-"):
        return []
    out = []
    for part in re.split(r"[;,]", text):
        m = re.fullmatch(r"\s*" + NUM + r"\s*-\s*(" + NUM[1:-1] + r"|end)\s*", part)
        if not m:
            raise ValueError(f"cannot read interval '{part}' in '{text}'")
        end = None if m.group(2) == "end" else float(m.group(2))
        out.append([float(m.group(1)), end])
    return out


def load_audit(path: str) -> pd.DataFrame:
    """Read the audit return CSV; keep rows that have a decision."""
    a = pd.read_csv(path, dtype=str).fillna("")
    a["decision"] = a["decision"].str.strip().str.lower()
    bad = sorted(set(a["decision"]) - set(AUDIT_DECISIONS) - {""})
    if bad:
        raise ValueError(f"unknown decision values in audit file: {bad}")
    return a


def apply_audit(records: pd.DataFrame, audit: pd.DataFrame,
                confirm_group_rule: bool = True,
                dispute_distortion_absent: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply audit verdicts on top of parser records. Returns (records_v2, change_log).

    * Only the audited fault of each row is changed (notes about other faults are ignored).
    * accept -> the proposed interval; localized -> the auditor's interval;
      absent -> no fault; present_unknown_time -> presence only; unknown -> stays masked.
    * Group rule: if every sampled row of the NISQA bare-number group was accepted,
      the remaining rows of that group get their proposed [X, end] interval.
    * dispute_distortion_absent: an 'absent' verdict for distortion on a clip whose
      original Distortion score is 1-4 contradicts the original annotator; such cells
      become 'disputed' (masked) instead of 'no fault' until a second listener checks them.
      The no-speech section is excluded (those clips were checked as a whole).
    * Clip durations from the audit fill 'end of clip' (None) for every record of that clip.
    """
    rec = records.copy()
    rec["intervals"] = rec["intervals"].map(lambda v: [list(x) for x in v])
    key = {(f, t): i for i, (f, t) in enumerate(zip(rec["file_id"], rec["fault"]))}
    log = []

    def set_cell(i, status, intervals, source, note, decision):
        before = (rec.at[i, "status"], format_intervals(rec.at[i, "intervals"]))
        rec.at[i, "status"] = status
        rec.at[i, "intervals"] = intervals
        rec.at[i, "label_source"] = source
        rec.at[i, "audit_note"] = note
        if rec.at[i, "conflict"]:
            rec.at[i, "conflict"] = False
            rec.at[i, "conflict_reason"] = "resolved by audit: " + rec.at[i, "conflict_reason"]
        log.append({"file_id": rec.at[i, "file_id"], "fault": rec.at[i, "fault"],
                    "decision": decision, "status_before": before[0], "intervals_before": before[1],
                    "status_after": status, "intervals_after": format_intervals(intervals),
                    "label_source": source, "note": note})

    decided = audit[audit["decision"] != ""]
    for r in decided.itertuples():
        i = key.get((r.file_id, r.fault))
        if i is None:
            continue
        d, note = r.decision, r.note
        if d == "accept":
            iv = parse_interval_text(r.proposed)
            if iv:
                set_cell(i, "audited", iv, "human_audit", note, d)
            else:
                set_cell(i, "ambiguous", [], "human_audit", "accept without proposal: " + note, d)
        elif d == "localized":
            iv = parse_interval_text(r.intervals)
            if iv:
                set_cell(i, "audited", iv, "human_audit", note, d)
            else:
                set_cell(i, "ambiguous", [], "human_audit", "localized without interval: " + note, d)
        elif d == "absent":
            score = rec.at[i, "score"]
            if (dispute_distortion_absent and r.fault == "distortion"
                    and not str(r.section).startswith("1") and pd.notna(score) and int(score) <= 4):
                set_cell(i, "disputed", [], "human_audit",
                         f"auditor: absent, original Distortion score {int(score)}: " + note, d)
            else:
                set_cell(i, "no_fault", [], "human_audit", note, d)
        elif d == "present_unknown_time":
            set_cell(i, "present_unknown_time", [], "human_audit", note, d)
        elif d == "unknown":
            st = rec.at[i, "status"]
            set_cell(i, st if st in REVIEW_STATUSES else "ambiguous", [], "human_audit",
                     "auditor: unknown: " + note, d)

    # group rule for the NISQA bare-number pauses
    group = audit[audit["section"].str.startswith("2")]
    sampled = group[group["must_listen"] == "yes"]
    if confirm_group_rule and len(sampled) and (sampled["decision"] == "accept").all():
        for r in group[group["decision"] == ""].itertuples():
            i = key.get((r.file_id, r.fault))
            iv = parse_interval_text(r.proposed)
            if i is not None and iv:
                set_cell(i, "rule_confirmed", iv, "group_rule",
                         f"NISQA bare-number rule confirmed by audit ({len(sampled)}/{len(sampled)} sampled)",
                         "group_rule")

    # durations from the audit -> fill 'end of clip'
    dur = {}
    for f, v in zip(audit["file_id"], audit["duration_s"]):
        try:
            dur[f] = float(v)
        except (TypeError, ValueError):
            pass
    for i in rec.index:
        d = dur.get(rec.at[i, "file_id"])
        if d is None:
            continue
        rec.at[i, "intervals"] = [[s, d if (e is None or e > d) else e]
                                  for s, e in rec.at[i, "intervals"] if s < d]

    rec["label_version"] = "v2"
    return rec, pd.DataFrame(log)


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------

def write_jsonl(records: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records.to_dict(orient="records"):
            rec["score"] = None if pd.isna(rec["score"]) else int(rec["score"])
            rec["conflict"] = bool(rec["conflict"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(csv_path: str, out_dir: str = "label_out", audit_path: Optional[str] = None,
        dispute_distortion_absent: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    records = parse_dataframe(df)
    os.makedirs(out_dir, exist_ok=True)
    if audit_path:
        records, log = apply_audit(records, load_audit(audit_path),
                                   dispute_distortion_absent=dispute_distortion_absent)
        log.to_csv(os.path.join(out_dir, "audit_changes.csv"), index=False, encoding="utf-8-sig")
    queue = review_queue(records, df)
    write_jsonl(records, os.path.join(out_dir, "parsed_labels.jsonl"))
    queue.to_csv(os.path.join(out_dir, "review_queue.csv"), index=False, encoding="utf-8-sig")
    return records, queue


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse QualiSpeech fault labels.")
    ap.add_argument("csv", help="path to train.csv / val.csv / test.csv")
    ap.add_argument("--out", default="label_out", help="output folder")
    ap.add_argument("--audit", default=None, help="filled audit return CSV -> labels v2")
    ap.add_argument("--trust-distortion-absent", action="action_store_true" if False else "store_true",
                    help="apply 'absent' distortion verdicts even when the original score was 1-4")
    args = ap.parse_args()
    records, queue = run(args.csv, args.out, args.audit,
                         dispute_distortion_absent=not args.trust_distortion_absent)
    print(summary(records).to_string())
    if args.audit:
        print("\nlabel source per status:")
        print(pd.crosstab(records["status"], records["label_source"]).to_string())
        print(f"changes written to {args.out}/audit_changes.csv")
    print(f"\nrows with score conflict: {int(records['conflict'].sum())}")
    print(f"review queue: {len(queue)} rows ({queue['file_id'].nunique()} files)")
    print(f"written to {args.out}/parsed_labels.jsonl and {args.out}/review_queue.csv")


if __name__ == "__main__":
    main()
