"""CTC decoders + Part B's four PDF metrics.

Reuses PartC's `greedy_ctc_decode` and `edit_distance` (already char-list aware).

The four PDF metrics, computed on `to_digits_only(pred)` vs `to_digits_only(gt)`:
  1. digit_accuracy = (1 - sum(I+D+S) / sum(N)) * 100  — aggregate, not per-amount mean
  2. pct_no_error    — %% amounts with edit_distance == 0
  3. pct_one_error   — %% amounts with edit_distance == 1
  4. pct_two_or_more — %% amounts with edit_distance >= 2

Also reports unstripped-string CER as diagnostic.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch

from PartC.decode import edit_distance, greedy_ctc_decode  # noqa: F401  (re-exported)
from PartB.courtesy_tokenizer import to_digits_only


def digit_metrics(preds: list[str], gts: list[str]) -> dict[str, float]:
    """Compute the four PDF metrics on digit-only strings."""
    p_digits = [to_digits_only(s) for s in preds]
    g_digits = [to_digits_only(s) for s in gts]

    edits_total = n_total = 0
    n_no = n_one = n_two_plus = 0

    for p, g in zip(p_digits, g_digits):
        ed = edit_distance(list(p), list(g))
        edits_total += ed
        n_total += len(g)
        if ed == 0:
            n_no += 1
        elif ed == 1:
            n_one += 1
        else:
            n_two_plus += 1

    n = max(1, len(p_digits))
    digit_acc = 100.0 * (1.0 - edits_total / max(1, n_total))

    # Diagnostic CER on unstripped strings (keeping '.' and '/').
    raw_edits = raw_total = 0
    for p, g in zip(preds, gts):
        raw_edits += edit_distance(list(p), list(g))
        raw_total += len(g)
    raw_cer = 100.0 * raw_edits / max(1, raw_total)

    return {
        "digit_accuracy": digit_acc,
        "digit_error_rate": 100.0 - digit_acc,
        "pct_no_error": 100.0 * n_no / n,
        "pct_one_error": 100.0 * n_one / n,
        "pct_two_or_more_errors": 100.0 * n_two_plus / n,
        "n_amounts": len(preds),
        "n_digits_total": n_total,
        "edits_total": edits_total,
        "raw_cer_unstripped": raw_cer,
    }


# ---------------------------------------------------------------------------
# Beam search (CTC prefix beam, no LM).
# ---------------------------------------------------------------------------


def _logsumexp(*xs: float) -> float:
    finite = [x for x in xs if x > -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(x - m) for x in finite))


def prefix_beam_decode(
    log_probs: torch.Tensor,
    idx_to_char: dict[int, str],
    blank: int = 0,
    beam_size: int = 10,
) -> list[str]:
    """CTC prefix-beam search. log_probs: (T, B, V). Returns list[str] length B.

    Standard prefix-beam — tracks (p_blank, p_nonblank) per prefix at each step.
    No external LM (we don't need one for digits).
    """
    T, B, V = log_probs.shape
    lp = log_probs.detach().cpu().numpy()
    out: list[str] = []
    for b in range(B):
        # Map prefix (tuple of int ids) -> (lp_blank, lp_nonblank).
        beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
        for t in range(T):
            new_beam: dict[tuple[int, ...], tuple[float, float]] = {}
            for prefix, (pb, pnb) in beam.items():
                # 1) extend by blank (no change in prefix).
                add_blank = lp[t, b, blank]
                new_pb = pb + add_blank
                new_pnb = pnb + add_blank
                npb, npnb = new_beam.get(prefix, (-math.inf, -math.inf))
                new_beam[prefix] = (
                    _logsumexp(npb, _logsumexp(new_pb, new_pnb)),
                    npnb,
                )
                # 2) extend by each non-blank symbol.
                for s in range(V):
                    if s == blank:
                        continue
                    p_s = lp[t, b, s]
                    if prefix and prefix[-1] == s:
                        # Same as last: must come from blank-end prefix to grow.
                        new_prefix = prefix + (s,)
                        npb2, npnb2 = new_beam.get(new_prefix, (-math.inf, -math.inf))
                        new_beam[new_prefix] = (npb2, _logsumexp(npnb2, pb + p_s))
                        # Also keep prefix unchanged via repeated symbol from non-blank end.
                        npb3, npnb3 = new_beam.get(prefix, (-math.inf, -math.inf))
                        new_beam[prefix] = (npb3, _logsumexp(npnb3, pnb + p_s))
                    else:
                        new_prefix = prefix + (s,)
                        npb2, npnb2 = new_beam.get(new_prefix, (-math.inf, -math.inf))
                        new_beam[new_prefix] = (
                            npb2,
                            _logsumexp(npnb2, _logsumexp(pb + p_s, pnb + p_s)),
                        )
            # Prune to top-K by total prob.
            scored = [(pref, _logsumexp(b1, b2)) for pref, (b1, b2) in new_beam.items()]
            scored.sort(key=lambda x: x[1], reverse=True)
            beam = {pref: new_beam[pref] for pref, _ in scored[:beam_size]}

        best_prefix = max(beam.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))[0]
        out.append("".join(idx_to_char.get(i, "") for i in best_prefix))
    return out
