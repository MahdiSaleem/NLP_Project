"""End-to-end verifier for Part D.

Loads:
  - Part B prediction file (`<file.tif> <digits>`)
  - Part C prediction file (`<file.tif> <space-joined PAWs>`)
  - Optional GT labels for blame attribution.

Emits:
  - PartD/runs/<name>/verification.txt — `<file> <courtesy_digits>
    <legal_digits> <MATCH|MISMATCH|UNPARSED>` lines.
  - PartD/runs/<name>/summary.json — aggregate counts and rates.

A row is MATCH iff parser(legal).value (integer part) == int(courtesy_digits).
UNPARSED if parser couldn't make sense of legal. Otherwise MISMATCH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PartD.arabic_numbers import parse_paw_list, parse_legal
from PartD.evaluate_parser import load_courtesy_raw, load_legal_raw


def load_predictions_b(path: Path) -> dict[str, str]:
    """{stem: digit_string}. Empty digits if YOLO missed."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        stem = Path(parts[0]).stem
        digits = parts[1].strip() if len(parts) > 1 else ""
        # Strip any non-digit fallback (parser may have emitted '.' etc.)
        digits = "".join(c for c in digits if c.isdigit())
        out[stem] = digits
    return out


def load_predictions_c(path: Path) -> dict[str, list[str]]:
    """{stem: paws_list}."""
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        stem = Path(parts[0]).stem
        paws = parts[1].split() if len(parts) > 1 else []
        out[stem] = paws
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partB", type=Path, required=True,
                    help="Path to predictions_courtesy.txt")
    ap.add_argument("--partC", type=Path, required=True,
                    help="Path to predictions_legal_paw.txt")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Directory to write verification.txt + summary.json")
    ap.add_argument("--split-list", type=Path, default=None,
                    help="Optional: restrict to image stems in this file (one per line).")
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path(r"C:\Users\qxawe\NLP_Project\Dataset"),
                    help="GT for blame attribution. Set --no-gt to skip.")
    ap.add_argument("--no-gt", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    preds_b = load_predictions_b(args.partB)
    preds_c = load_predictions_c(args.partC)
    print(f"PartB predictions: {len(preds_b)}, PartC predictions: {len(preds_c)}")

    # Optional split restriction.
    keep: set[str] | None = None
    if args.split_list:
        keep = {Path(line.strip()).stem for line in args.split_list.read_text().splitlines() if line.strip()}
        print(f"Restricting to {len(keep)} stems from {args.split_list}")

    # Optional GT for blame.
    gt_courtesy: dict[str, str] = {}
    gt_legal: dict[str, str] = {}
    if not args.no_gt:
        try:
            gt_courtesy = load_courtesy_raw(args.dataset_dir / "CourtesyAmounts_raw")
            gt_legal = load_legal_raw(args.dataset_dir / "LegalAmounts_raw_text")
            print(f"GT loaded: legal={len(gt_legal)}, courtesy={len(gt_courtesy)}")
        except Exception as e:
            print(f"WARN: GT load failed ({e}). Proceeding without blame attribution.")

    stems = sorted(set(preds_b) & set(preds_c))
    if keep:
        stems = [s for s in stems if s in keep]

    n = match = mismatch = unparsed = 0
    blame_match_only = legal_blame = courtesy_blame = both_blame = 0
    rows: list[dict] = []
    out_txt = args.out_dir / "verification.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        for stem in stems:
            digits = preds_b.get(stem, "")
            paws = preds_c.get(stem, [])
            res = parse_paw_list(paws) if paws else parse_legal("")
            n += 1
            legal_digits = str(res.value) if (res.valid and res.value is not None) else ""
            label: str
            if not legal_digits:
                label = "UNPARSED"
                unparsed += 1
            elif legal_digits == digits and digits != "":
                label = "MATCH"
                match += 1
            else:
                label = "MISMATCH"
                mismatch += 1
            f.write(f"{stem}.tif {digits} {legal_digits} {label}\n")

            row = {
                "stem": stem, "courtesy": digits, "legal": legal_digits,
                "label": label, "leftover": res.leftover[:5],
            }

            # Blame attribution if GT available.
            if gt_courtesy and gt_legal and stem in gt_courtesy and stem in gt_legal:
                gt_int = int(gt_courtesy[stem])
                gt_str = str(gt_int)
                courtesy_ok = (digits == gt_str)
                legal_ok = (legal_digits == gt_str) if legal_digits else False
                row["gt"] = gt_str
                row["courtesy_ok"] = courtesy_ok
                row["legal_ok"] = legal_ok
                if courtesy_ok and legal_ok:
                    blame_match_only += 1
                elif courtesy_ok and not legal_ok:
                    legal_blame += 1
                elif legal_ok and not courtesy_ok:
                    courtesy_blame += 1
                else:
                    both_blame += 1
            rows.append(row)

    summary = {
        "partB": str(args.partB),
        "partC": str(args.partC),
        "split": str(args.split_list) if args.split_list else "all",
        "n_total": n,
        "n_match": match,
        "n_mismatch": mismatch,
        "n_unparsed": unparsed,
        "match_rate": 100.0 * match / max(1, n),
        "match_rate_among_parsed": 100.0 * match / max(1, n - unparsed),
        "blame": {
            "both_correct_and_match": blame_match_only,
            "legal_recognizer_blame": legal_blame,
            "courtesy_recognizer_blame": courtesy_blame,
            "both_blame": both_blame,
        } if gt_courtesy else None,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {out_txt} + summary.json + rows.json")
    print(f"\nTotal: {n}")
    print(f"  MATCH:    {match} ({100*match/max(1,n):.2f}%)")
    print(f"  MISMATCH: {mismatch} ({100*mismatch/max(1,n):.2f}%)")
    print(f"  UNPARSED: {unparsed} ({100*unparsed/max(1,n):.2f}%)")
    if gt_courtesy and gt_legal:
        print(f"\nBlame attribution (rows w/ GT):")
        print(f"  both correct & match:   {blame_match_only}")
        print(f"  legal recognizer wrong: {legal_blame}")
        print(f"  courtesy wrong:         {courtesy_blame}")
        print(f"  both wrong:             {both_blame}")


if __name__ == "__main__":
    main()
