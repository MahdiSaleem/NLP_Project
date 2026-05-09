"""Calibrate the rule-based parser on GT pairs.

Loads:
  - GT legal text from Dataset/LegalAmounts_raw_text/
  - GT courtesy digits from Dataset/CourtesyAmounts_raw/
  - Falls back to tokenized variants when raw missing.

Intersects IDs and reports:
  - parser_accuracy: % of pairs where parse_legal(text).value == int(courtesy)
  - unparseable_pct
  - mismatch examples (top 30)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PartD.arabic_numbers import parse_legal


def load_legal_raw(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for fp in sorted(path.glob("*.txt")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            raw_id, text = parts
            if raw_id.startswith("L"):
                raw_id = raw_id[1:]
            stem = Path(raw_id).stem
            if text.strip():
                out[stem] = text.strip()
    return out


def load_courtesy_raw(path: Path) -> dict[str, str]:
    """Returns {stem: integer-part-as-str}. Drops cents."""
    out: dict[str, str] = {}
    for fp in sorted(path.glob("*.txt")):
        for line in fp.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            raw_id, amt = parts
            if raw_id.startswith("C"):
                raw_id = raw_id[1:]
            stem = Path(raw_id).stem
            int_part = amt.split(".")[0].strip()
            if int_part.lstrip("-").isdigit():
                out[stem] = int_part
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path(r"C:\Users\qxawe\NLP_Project\Dataset"))
    ap.add_argument("--out", type=Path, default=Path("PartD/runs/parser_calib.json"))
    ap.add_argument("--show-mismatches", type=int, default=30)
    args = ap.parse_args()

    legal = load_legal_raw(args.dataset_dir / "LegalAmounts_raw_text")
    courtesy = load_courtesy_raw(args.dataset_dir / "CourtesyAmounts_raw")
    common = sorted(set(legal) & set(courtesy))
    print(f"GT legal entries: {len(legal)}, GT courtesy entries: {len(courtesy)}, common: {len(common)}")

    n = 0
    n_match = 0
    n_unparsed = 0
    mismatches: list[dict] = []

    for stem in common:
        text = legal[stem]
        gt_int = int(courtesy[stem])
        res = parse_legal(text)
        n += 1
        if not res.valid or res.value is None:
            n_unparsed += 1
            mismatches.append({
                "stem": stem, "text": text, "parsed": None,
                "gt": gt_int, "leftover": res.leftover, "normalized": res.normalized,
            })
            continue
        if res.value == gt_int:
            n_match += 1
        else:
            mismatches.append({
                "stem": stem, "text": text, "parsed": res.value,
                "gt": gt_int, "leftover": res.leftover, "normalized": res.normalized,
            })

    acc = 100.0 * n_match / max(1, n)
    unparsed_rate = 100.0 * n_unparsed / max(1, n)
    print(f"Parser accuracy: {acc:.2f}%  ({n_match}/{n})")
    print(f"Unparseable:     {unparsed_rate:.2f}%  ({n_unparsed}/{n})")
    print(f"Mismatched:      {len(mismatches) - n_unparsed}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_pairs": n,
        "n_match": n_match,
        "n_unparsed": n_unparsed,
        "parser_accuracy": acc,
        "unparseable_rate": unparsed_rate,
        "mismatches": mismatches,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    print(f"\nTop {args.show_mismatches} examples (id only — see JSON for text):")
    for m in mismatches[:args.show_mismatches]:
        p = m["parsed"] if m["parsed"] is not None else "UNPARSED"
        print(f"  {m['stem']}: parsed={p}  gt={m['gt']}")


if __name__ == "__main__":
    main()
