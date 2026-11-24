"""Kiểm thử script nhận bản phát hành LO_V3_1 trên một bản phát hành giả lập nhỏ."""
import json

import numpy as np
import pandas as pd

import check_lo_v3_1_release as chk


def tiny_release(seed=0):
    rng = np.random.default_rng(seed)
    manifest = pd.DataFrame({
        "window_id": ["W1"] * 4, "split": ["train", "train", "validation", "test"],
        "offering_id": ["o1", "o2", "o3", "o4"], "timeline_source": ["course_specific_timeline"] * 4,
        "offering_end_date": pd.to_datetime(["2020-05-01", "2020-07-31", "2020-09-01", "2020-09-10"])})
    rows = []
    for split, offering, n in (("train", "o1", 40), ("train", "o2", 40), ("validation", "o3", 20), ("test", "o4", 20)):
        for i in range(n):
            score = float(rng.uniform(0, 100))
            label = "E" if score >= 85 else "G" if score >= 60 else "I/D"
            rows.append({"window": "W1", "split": split, "enrollment_id": f"{offering}-{i}",
                         "offering_id": offering, "timeline_source": "course_specific_timeline",
                         "LO_performance_label_3": label, "performance_score": score})
    frame = pd.DataFrame(rows)
    for p in range(1, 5):
        frame[f"watch_count_P{p}"] = rng.normal(size=len(frame))
        frame[f"phase_available_P{p}"] = 1
        frame[f"primary_risk_set_P{p}"] = 1
    tv = frame[frame["split"] != "test"].reset_index(drop=True)
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    prefixes = {}
    for k, phase in enumerate(chk.PHASES, start=1):
        view = test.copy()
        for p in range(1, 5):
            view[f"phase_available_P{p}"] = int(p <= k)
            if p > k:
                for column in (f"watch_count_P{p}", f"primary_risk_set_P{p}"):
                    view[column] = np.nan
        view["prediction_phase"] = phase
        prefixes[phase] = view
    return manifest, tv, prefixes


def test_identity_is_checked_exactly():
    assert chk.check_identity(dict(chk.EXPECTED_IDENTITY)) == []
    wrong = dict(chk.EXPECTED_IDENTITY, label_rule_version="lo_operational_performance_v1")
    assert len(chk.check_identity(wrong)) == 1


def test_clean_release_passes_hard_checks():
    manifest, tv, prefixes = tiny_release()
    assert chk.check_offerings(manifest) == []
    assert chk.check_grain(tv, prefixes["P4"]) == []
    assert chk.check_labels(tv) == []
    assert chk.check_test_prefixes(prefixes) == []


def test_enrollment_in_two_splits_and_duplicates_fail():
    manifest, tv, prefixes = tiny_release()
    moved = prefixes["P4"].iloc[[0]].assign(split="validation")
    assert chk.check_grain(pd.concat([tv, moved]), prefixes["P4"])
    assert chk.check_grain(pd.concat([tv, tv.iloc[[0]]]), prefixes["P4"])


def test_offering_in_two_splits_or_late_train_label_fails():
    manifest, _, _ = tiny_release()
    leak = pd.concat([manifest, manifest.iloc[[0]].assign(split="test")])
    assert chk.check_offerings(leak)
    late = manifest.copy()
    late.loc[1, "offering_end_date"] = pd.Timestamp("2020-12-31")
    assert chk.check_offerings(late)


def test_band_violation_fails():
    _, tv, _ = tiny_release()
    bad = tv.copy()
    bad.loc[0, "performance_score"] = 90.0
    bad.loc[0, "LO_performance_label_3"] = "I/D"
    assert chk.check_labels(bad)


def test_future_value_in_prefix_fails():
    _, _, prefixes = tiny_release()
    leaked = {k: v.copy() for k, v in prefixes.items()}
    leaked["P1"].loc[0, "watch_count_P3"] = 1.0
    assert any("watch_count_P3" in f for f in chk.check_test_prefixes(leaked))


def test_cli_end_to_end(tmp_path):
    manifest, tv, prefixes = tiny_release()
    (tmp_path / "id.json").write_text(json.dumps(chk.EXPECTED_IDENTITY))
    (tmp_path / "manifest").mkdir()
    manifest.to_parquet(tmp_path / "manifest" / "part-0.parquet")
    (tmp_path / "tv").mkdir()
    tv.to_parquet(tmp_path / "tv" / "part-0.parquet")
    for phase, frame in prefixes.items():
        (tmp_path / "test" / phase).mkdir(parents=True)
        frame.to_parquet(tmp_path / "test" / phase / "part-0.parquet")
    code = chk.main(["--identity", str(tmp_path / "id.json"), "--split-manifest", str(tmp_path / "manifest"),
                     "--train-validation", str(tmp_path / "tv"), "--test-prefix-root", str(tmp_path / "test"),
                     "--report", str(tmp_path / "report.json")])
    report = json.loads((tmp_path / "report.json").read_text())
    assert code == 0 and report["fails"] == []
    assert report["warns"]                                   # dữ liệu giả lập không khớp số của registry
    assert report["test_primary_risk_set_share"]["P1"] == 1.0
