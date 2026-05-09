"""BONUS — mutual correction of courtesy ⇄ legal predictions.

Strategy: blame attribution (in `verify.py`) showed that on the full 1800-
check set, the legal recognizer is wrong 5x more often than the courtesy
recognizer (548 legal_blame vs 107 courtesy_blame). So when the two sides
disagree by a small margin, trust courtesy.

Two strategies are reported:
  A) `snap_legal_to_courtesy` (default): when parsed_legal differs from
     courtesy_digits by edit distance == 1, replace parsed_legal with
     courtesy_digits. Never overwrites courtesy. Backfills legal from
     courtesy when legal is UNPARSED.
  B) `legal_clean_to_courtesy` (toggle): same as A but additionally, when
     parsed_legal came from a clean parse (no leftover, no cents
     confusion), trust legal and overwrite courtesy. Reported as a
     diagnostic — usually worse on this dataset because "clean parse" is
     a weak signal of correctness.

Reports:
  - Pre/post legal-recognizer accuracy (legal_digits == GT)
  - Pre/post courtesy-recognizer accuracy (courtesy_digits == GT)
  - Pre/post match-rate (legal == courtesy)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PartD.arabic_numbers import parse_paw_list
from PartD.evaluate_parser import load_courtesy_raw
from PartD.verify import load_predictions_b, load_predictions_c
from PartC.decode import edit_distance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partB", type=Path, required=True)
    ap.add_argument("--partC", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split-list", type=Path, default=None)
    ap.add_argument("--dataset-dir", type=Path,
                    default=Path(r"C:\Users\qxawe\NLP_Project\Dataset"))
    ap.add_argument("--strategy", choices=["snap_legal_to_courtesy",
                                           "legal_clean_to_courtesy"],
                    default="snap_legal_to_courtesy")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    preds_b = load_predictions_b(args.partB)
    preds_c = load_predictions_c(args.partC)
    gt_courtesy = load_courtesy_raw(args.dataset_dir / "CourtesyAmounts_raw")

    keep: set[str] | None = None
    if args.split_list:
        keep = {Path(line.strip()).stem for line in args.split_list.read_text().splitlines() if line.strip()}

    stems = sorted(set(preds_b) & set(preds_c))
    if keep:
        stems = [s for s in stems if s in keep]

    n = 0
    legal_ok_pre = courtesy_ok_pre = match_pre = 0
    legal_ok_post = courtesy_ok_post = match_post = 0
    n_corrected_legal = n_corrected_courtesy = 0
    rows: list[dict] = []

    for stem in stems:
        if stem not in gt_courtesy:
            continue
        n += 1
        c_orig = preds_b.get(stem, "")
        paws = preds_c.get(stem, [])
        res = parse_paw_list(paws) if paws else None
        l_orig = str(res.value) if (res and res.valid and res.value is not None) else ""
        gt = str(int(gt_courtesy[stem]))

        # Pre-correction.
        c_pre_ok = c_orig == gt
        l_pre_ok = l_orig == gt
        m_pre_ok = bool(l_orig) and l_orig == c_orig
        legal_ok_pre += int(l_pre_ok)
        courtesy_ok_pre += int(c_pre_ok)
        match_pre += int(m_pre_ok)

        # Mutual correction.
        c_new, l_new = c_orig, l_orig
        legal_clean = bool(res and res.valid and not res.leftover and res.cents is None)
        if l_orig and c_orig and l_orig != c_orig:
            ed = edit_distance(list(l_orig), list(c_orig))
            if ed == 1:
                if args.strategy == "legal_clean_to_courtesy" and legal_clean:
                    # Aggressive: trust clean-parsed legal — overwrite courtesy.
                    c_new = l_orig
                    n_corrected_courtesy += 1
                else:
                    # Default: trust courtesy — overwrite legal.
                    l_new = c_orig
                    n_corrected_legal += 1
        elif not l_orig and c_orig:
            # Legal unparsed — fill from courtesy.
            l_new = c_orig
            n_corrected_legal += 1

        c_post_ok = c_new == gt
        l_post_ok = l_new == gt
        m_post_ok = bool(l_new) and l_new == c_new
        legal_ok_post += int(l_post_ok)
        courtesy_ok_post += int(c_post_ok)
        match_post += int(m_post_ok)

        rows.append({
            "stem": stem, "gt": gt,
            "courtesy_orig": c_orig, "legal_orig": l_orig,
            "courtesy_new": c_new, "legal_new": l_new,
            "legal_clean_parse": legal_clean,
            "match_pre": m_pre_ok, "match_post": m_post_ok,
        })

    def pct(x: int) -> float:
        return 100.0 * x / max(1, n)

    summary = {
        "n_with_gt": n,
        "legal_acc_pre": pct(legal_ok_pre),
        "legal_acc_post": pct(legal_ok_post),
        "legal_acc_lift": pct(legal_ok_post) - pct(legal_ok_pre),
        "courtesy_acc_pre": pct(courtesy_ok_pre),
        "courtesy_acc_post": pct(courtesy_ok_post),
        "courtesy_acc_lift": pct(courtesy_ok_post) - pct(courtesy_ok_pre),
        "match_rate_pre": pct(match_pre),
        "match_rate_post": pct(match_post),
        "match_rate_lift": pct(match_post) - pct(match_pre),
        "n_corrected_legal": n_corrected_legal,
        "n_corrected_courtesy": n_corrected_courtesy,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "rows.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nMutual correction over {n} stems with GT:")
    print(f"  legal_acc:    {summary['legal_acc_pre']:.2f}% -> {summary['legal_acc_post']:.2f}%  ({summary['legal_acc_lift']:+.2f} pp)")
    print(f"  courtesy_acc: {summary['courtesy_acc_pre']:.2f}% -> {summary['courtesy_acc_post']:.2f}%  ({summary['courtesy_acc_lift']:+.2f} pp)")
    print(f"  match_rate:   {summary['match_rate_pre']:.2f}% -> {summary['match_rate_post']:.2f}%  ({summary['match_rate_lift']:+.2f} pp)")
    print(f"  legal corrected: {n_corrected_legal}, courtesy corrected: {n_corrected_courtesy}")


if __name__ == "__main__":
    main()
