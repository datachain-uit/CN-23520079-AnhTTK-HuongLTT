"""Kiểm tra bản phát hành LO_V3_1 nhận từ CD3 trước khi dùng. Chỉ đọc, không sửa dữ liệu.

FAIL (dừng, báo giảng viên): định danh manifest; grain; offering nằm ở nhiều split;
thứ tự nhãn tau_w; test prefix lộ cột tương lai; P1-P4 khác tập enrollment; nhãn sai band.
WARN (ghi vào báo cáo tuần): số enrollment theo lớp khác registry CD3 (Bảng C4, C9).

python check_lo_v3_1_release.py --identity lo_v3_1_release_manifest.json \
    --split-manifest <dir> --train-validation <dir> --test-prefix-root <dir> [--window W1]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED_IDENTITY = {
    "task": "LO", "release_id": "LO_V3_1",
    "split_registry_id": "split_registry_v3_1_scored_signal_excluded_overlap_audit_v1",
    "split_version": "v3_1", "phase_version": "wide_prefix_v3_1",
    "feature_dictionary_version": "schema_from_phase_views_v3_1",
    "label_rule_version": "lo_final_score_catalog_normalized_v3_1",
    "label_threshold_set": "CATALOG_NORMALIZED_PRIMARY_V3_1__EXCLUDE_NO_SCORED_SIGNAL",
}
LABELS = ("I/D", "G", "E")
BLOCK_CLASSES = {  # LO/registries.py, tasks.LO.block_classes (c0 = I/D, c1 = G, c2 = E)
    "A": (1782182, 4720, 655), "B": (77306, 532, 252), "C": (77066, 525, 1000),
    "D": (200812, 23, 7), "E": (200810, 23, 11), "F": (129283, 94, 49), "G": (128965, 1079, 29)}
WINDOWS = {"W1": (("A",), "B", "C"), "W2": (("A", "B", "C"), "D", "E"),
           "W3": (("A", "B", "C", "D", "E"), "F", "G")}
PHASES = ("P1", "P2", "P3", "P4")


def expected_counts() -> dict[tuple[str, str], dict[str, int]]:
    result = {}
    for window, (train, validation, test) in WINDOWS.items():
        for split, blocks in (("train", train), ("validation", (validation,)), ("test", (test,))):
            result[(window, split)] = {label: sum(BLOCK_CLASSES[b][i] for b in blocks)
                                       for i, label in enumerate(LABELS)}
    return result


def check_identity(identity: dict) -> list[str]:
    return [f"identity {k}: {identity.get(k)!r} != {v!r}" for k, v in EXPECTED_IDENTITY.items()
            if identity.get(k) != v]


def check_grain(tv: pd.DataFrame, test_p4: pd.DataFrame) -> list[str]:
    fails = []
    rows = pd.concat([tv[["window", "split", "enrollment_id"]], test_p4[["window", "split", "enrollment_id"]]])
    if rows.duplicated().any():
        fails.append(f"grain: {int(rows.duplicated().sum())} duplicated (window, split, enrollment_id) rows")
    splits = rows.groupby(["window", "enrollment_id"])["split"].nunique()
    if (splits > 1).any():
        fails.append(f"grain: {int((splits > 1).sum())} enrollments in more than one split of a window")
    return fails


def check_offerings(split_manifest: pd.DataFrame) -> list[str]:
    fails = []
    unit = ["window_id", "offering_id", "timeline_source"]
    multi = split_manifest.groupby(unit)["split"].nunique()
    if (multi > 1).any():
        fails.append(f"split: {int((multi > 1).sum())} offering units in more than one split")
    ends = pd.to_datetime(split_manifest["offering_end_date"])
    for window, frame in split_manifest.assign(_end=ends).groupby("window_id"):
        tau = frame.loc[frame["split"] == "train", "_end"].max()
        first_eval = frame.loc[frame["split"] != "train", "_end"].min()
        if not tau <= first_eval:
            fails.append(f"split {window}: tau_w {tau} > earliest validation/test offering end {first_eval}")
    return fails


def check_labels(frame: pd.DataFrame) -> list[str]:
    fails = []
    bad = ~frame["LO_performance_label_3"].isin(LABELS)
    if bad.any():
        fails.append(f"label: {int(bad.sum())} rows with label outside {LABELS}")
    if "performance_score" in frame:
        s = frame["performance_score"].astype(float)
        band = pd.Series("I/D", index=frame.index).mask(s >= 60, "G").mask(s >= 85, "E")
        if ((s < 0) | (s > 100)).any():
            fails.append("label: performance_score outside [0, 100]")
        if (band != frame["LO_performance_label_3"]).any():
            fails.append(f"label: {int((band != frame['LO_performance_label_3']).sum())} rows violate the 60/85 bands")
    return fails


def check_test_prefixes(prefixes: dict[str, pd.DataFrame]) -> list[str]:
    fails = []
    reference = prefixes["P4"].set_index(["window", "enrollment_id"])["LO_performance_label_3"].sort_index()
    for k, phase in enumerate(PHASES, start=1):
        frame = prefixes[phase]
        keyed = frame.set_index(["window", "enrollment_id"])["LO_performance_label_3"].sort_index()
        if not keyed.index.equals(reference.index) or not keyed.equals(reference):
            fails.append(f"test {phase}: enrollment set or labels differ from P4")
        for column in frame.columns:
            if column.startswith("phase_available_P"):
                expected = int(int(column[-1]) <= k)
                if not (frame[column] == expected).all():
                    fails.append(f"test {phase}: {column} != {expected}")
            elif column.endswith(("_P1", "_P2", "_P3", "_P4")) and int(column[-1]) > k \
                    and not column.startswith(("primary_risk_set_", "label_available_by_cutoff_",
                                               "temporal_strict_", "cutoff_time_")):
                if frame[column].notna().any():
                    fails.append(f"test {phase}: future column {column} has values")
    return fails


def compare_counts(tv: pd.DataFrame, test_p4: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = pd.concat([tv[["window", "split", "LO_performance_label_3"]],
                      test_p4[["window", "split", "LO_performance_label_3"]]])
    observed = rows.groupby(["window", "split", "LO_performance_label_3"]).size()
    table, warns = [], []
    for (window, split), expected in expected_counts().items():
        if window not in set(rows["window"]):
            continue
        for label in LABELS:
            got = int(observed.get((window, split, label), 0))
            table.append({"window": window, "split": split, "label": label, "observed": got,
                          "registry": expected[label]})
            if got != expected[label]:
                warns.append(f"count {window}/{split}/{label}: {got} != registry {expected[label]}")
    return pd.DataFrame(table), warns


def _read(path: Path, columns=None, window=None) -> pd.DataFrame:
    import pyarrow.dataset as ds
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    names = set(dataset.schema.names)
    columns = None if columns is None else [c for c in columns if c in names]
    window_col = "window" if "window" in names else "window_id"
    flt = None if window is None else (ds.field(window_col) == window)
    return dataset.to_table(columns=columns, filter=flt).to_pandas()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--split-manifest", type=Path, required=True)
    ap.add_argument("--train-validation", type=Path, required=True)
    ap.add_argument("--test-prefix-root", type=Path, required=True)
    ap.add_argument("--window", choices=tuple(WINDOWS), default=None)
    ap.add_argument("--report", type=Path, default=Path("lo_v3_1_intake_report.json"))
    args = ap.parse_args(argv)
    keys = ["window", "split", "enrollment_id", "offering_id", "timeline_source",
            "LO_performance_label_3", "performance_score"]
    fails = check_identity(json.loads(args.identity.read_text(encoding="utf-8")))
    fails += check_offerings(_read(args.split_manifest, ["window_id", "split", "offering_id", "timeline_source",
                                                         "offering_end_date"], args.window))
    tv = _read(args.train_validation, keys, args.window)
    prefixes = {p: _read(args.test_prefix_root / p, None, args.window) for p in PHASES}
    fails += check_grain(tv, prefixes["P4"]) + check_labels(tv) + check_labels(prefixes["P4"])
    fails += check_test_prefixes(prefixes)
    table, warns = compare_counts(tv, prefixes["P4"])
    risk = {p: float(prefixes[p][f"primary_risk_set_{p}"].mean()) for p in PHASES
            if f"primary_risk_set_{p}" in prefixes[p]}
    report = {"fails": fails, "warns": warns, "counts": table.to_dict("records"),
              "test_primary_risk_set_share": risk}
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(table.to_string(index=False))
    for line in fails:
        print("FAIL", line)
    for line in warns:
        print("WARN", line)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
