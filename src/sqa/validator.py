from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

REQUIRED_FAULT_TYPES = {"noise", "distortion", "unnatural_pause"}


@dataclass
class ValidationError:

    file_id: str
    field: str
    error_type: str
    message: str

    def __str__(self) -> str:
        return f"[{self.file_id}] {self.field} ({self.error_type}): {self.message}"


def validate_submission_entry(entry: dict, duration_lookup: dict, collect_errors: list | None = None) -> bool:

    def fail(field_name: str, error_type: str, message: str) -> bool:
        if collect_errors is not None:
            collect_errors.append(ValidationError(entry.get("file_id", "?"), field_name, error_type, message))
            return False
        raise AssertionError(f"[{entry.get('file_id', '?')}] {field_name}: {message}")

    ok = True
    fid = entry.get("file_id")
    if not isinstance(fid, str) or fid not in duration_lookup:
        ok = fail("file_id", "invalid", f"file_id không hợp lệ hoặc không có trong test set: {fid!r}")
        if collect_errors is None:
            return False

    q = entry.get("quality_score")
    if isinstance(q, bool) or not isinstance(q, (int, float)):
        ok = fail("quality_score", "type", f"phải là số, không phải bool: {q!r}") and ok
    elif math.isnan(q) or math.isinf(q):
        ok = fail("quality_score", "value", f"NaN/Infinity: {q!r}") and ok
    elif not (1.0 <= q <= 5.0):
        ok = fail("quality_score", "range", f"ngoài [1,5]: {q!r}") and ok

    faults = entry.get("faults")
    if not isinstance(faults, dict) or set(faults.keys()) != REQUIRED_FAULT_TYPES:
        ok = fail("faults", "schema", f"thiếu/thừa key hoặc không phải dict: {faults!r}") and ok
        if collect_errors is None:
            return False
    else:
        dur = duration_lookup.get(fid, float("inf"))
        for ft, intervals in faults.items():
            if not isinstance(intervals, list):
                ok = fail(f"faults.{ft}", "type", f"phải là list: {intervals!r}") and ok
                continue
            prev_end = -1.0
            for iv in intervals:
                if not isinstance(iv, list) or len(iv) != 2:
                    ok = fail(f"faults.{ft}", "interval_shape", f"phải có đúng 2 phần tử: {iv!r}") and ok
                    continue
                s, e = iv
                if (
                    isinstance(s, bool)
                    or isinstance(e, bool)
                    or not isinstance(s, (int, float))
                    or not isinstance(e, (int, float))
                ):
                    ok = fail(f"faults.{ft}", "interval_type", f"start/end phải là số: {iv!r}") and ok
                    continue
                if math.isnan(s) or math.isnan(e) or math.isinf(s) or math.isinf(e):
                    ok = fail(f"faults.{ft}", "interval_value", f"NaN/Infinity: {iv!r}") and ok
                    continue
                if not (s < e):
                    ok = fail(f"faults.{ft}", "interval_order", f"rỗng/đảo ngược: {iv!r}") and ok
                    continue
                if not (0 <= s <= dur and 0 <= e <= dur):
                    ok = fail(f"faults.{ft}", "interval_bounds", f"vượt duration={dur}: {iv!r}") and ok
                    continue
                if s < prev_end:
                    ok = fail(f"faults.{ft}", "interval_overlap", f"chưa sort/chồng lấn chưa merge: {iv!r}") and ok
                prev_end = e
    return ok


def validate_submission(entries: list[dict], duration_lookup: dict, strict: bool = True):

    errors: list = [] if not strict else None
    seen = set()

    for entry in entries:
        if strict:
            validate_submission_entry(entry, duration_lookup, collect_errors=None)
        else:
            validate_submission_entry(entry, duration_lookup, collect_errors=errors)
        fid = entry.get("file_id")
        if fid in seen:
            msg = f"file_id trùng lặp: {fid}"
            if strict:
                raise AssertionError(msg)
            errors.append(ValidationError(fid, "file_id", "duplicate", msg))
        seen.add(fid)

    expected = set(duration_lookup.keys())
    missing, extra = expected - seen, seen - expected
    if missing or extra:
        msg = f"Thiếu {len(missing)} (vd {list(missing)[:5]}), thừa {len(extra)} (vd {list(extra)[:5]})"
        if strict:
            raise AssertionError(msg)
        errors.append(ValidationError("<submission>", "file_set", "mismatch", msg))

    reloaded = json.loads(json.dumps(entries))
    if strict:
        assert reloaded == entries, "Dữ liệu thay đổi sau round-trip JSON — kiểm tra kiểu dữ liệu không chuẩn."
    elif reloaded != entries:
        errors.append(ValidationError("<submission>", "json", "round_trip_mismatch", "Dữ liệu thay đổi sau round-trip JSON"))

    return errors
