import json
import math

import pytest

from sqa.validator import validate_submission, validate_submission_entry

DURS = {"a.wav": 2.0, "b.wav": 3.0}


def good_entry(file_id="a.wav"):
    return {
        "file_id": file_id,
        "quality_score": 3.5,
        "faults": {"noise": [[0.1, 0.5]], "distortion": [], "unnatural_pause": []},
    }


def test_valid_entry_passes():
    assert validate_submission_entry(good_entry(), DURS) is True


def test_empty_faults_everywhere_passes():
    e = good_entry()
    e["faults"] = {"noise": [], "distortion": [], "unnatural_pause": []}
    assert validate_submission_entry(e, DURS) is True


def test_touching_intervals_pass():
    e = good_entry()
    e["faults"]["noise"] = [[0.0, 0.5], [0.5, 1.0]]
    assert validate_submission_entry(e, DURS) is True


def test_quality_score_bool_rejected():
    errors = []
    e = good_entry()
    e["quality_score"] = True
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.field == "quality_score" and err.error_type == "type" for err in errors)


def test_quality_score_out_of_range_rejected():
    errors = []
    e = good_entry()
    e["quality_score"] = 6.0
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "range" for err in errors)


def test_interval_start_as_string_does_not_raise_typeerror():
    errors = []
    e = good_entry()
    e["faults"]["noise"] = [["abc", 0.5]]
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "interval_type" for err in errors)


def test_nan_interval_rejected():
    errors = []
    e = good_entry()
    e["faults"]["noise"] = [[float("nan"), 0.5]]
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "interval_value" for err in errors)


def test_infinity_interval_rejected():
    errors = []
    e = good_entry()
    e["faults"]["noise"] = [[0.0, float("inf")]]
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "interval_value" for err in errors)


def test_reversed_interval_rejected():
    errors = []
    e = good_entry()
    e["faults"]["noise"] = [[0.5, 0.1]]
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "interval_order" for err in errors)


def test_interval_beyond_duration_rejected():
    errors = []
    e = good_entry()
    e["faults"]["noise"] = [[0.0, 5.0]]  # duration của a.wav chỉ 2.0
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "interval_bounds" for err in errors)


def test_overlapping_unmerged_intervals_rejected():
    errors = []
    e = good_entry()
    e["faults"]["noise"] = [[0.0, 1.0], [0.5, 1.5]]  # chồng lấn, chưa merge
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "interval_overlap" for err in errors)


def test_missing_faults_key_rejected():
    errors = []
    e = good_entry()
    del e["faults"]["distortion"]
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.field == "faults" and err.error_type == "schema" for err in errors)


def test_extra_faults_key_rejected():
    errors = []
    e = good_entry()
    e["faults"]["extra_fault"] = []
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok


def test_interval_not_list_of_two_rejected():
    errors = []
    e = good_entry()
    e["faults"]["noise"] = [[0.1, 0.2, 0.3]]
    ok = validate_submission_entry(e, DURS, collect_errors=errors)
    assert not ok
    assert any(err.error_type == "interval_shape" for err in errors)


def test_strict_mode_raises_on_first_error():
    e = good_entry()
    e["quality_score"] = True
    with pytest.raises(AssertionError):
        validate_submission_entry(e, DURS)



def test_duplicate_file_id_detected_non_strict():
    entries = [good_entry("a.wav"), good_entry("a.wav")]
    errors = validate_submission(entries, DURS, strict=False)
    assert any(err.error_type == "duplicate" for err in errors)


def test_missing_file_detected_non_strict():
    entries = [good_entry("a.wav")]  # thiếu b.wav
    errors = validate_submission(entries, DURS, strict=False)
    assert any(err.error_type == "mismatch" for err in errors)


def test_valid_full_submission_strict_mode_no_raise():
    entries = [good_entry("a.wav"), good_entry("b.wav")]
    result = validate_submission(entries, DURS, strict=True)
    assert result is None


def test_json_round_trip_matches():
    entries = [good_entry("a.wav"), good_entry("b.wav")]
    reloaded = json.loads(json.dumps(entries))
    assert reloaded == entries
