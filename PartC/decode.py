"""CTC decoders + Levenshtein metrics for Part C.

`greedy_ctc_decode`  - argmax + collapse-repeats + drop-blanks.
`edit_distance`      - dynamic-programming Levenshtein (chars or tokens).
`cer_wer`            - aggregate (I+D+S)/N over a list of (pred, gt) pairs.
"""
from __future__ import annotations

import torch


def greedy_ctc_decode(log_probs: torch.Tensor, idx_to_char: dict[int, str], blank: int = 0) -> list[str]:
    """log_probs: (T, B, V). Returns list of decoded strings, length B."""
    preds = log_probs.argmax(dim=2).transpose(0, 1)  # (B, T)
    out: list[str] = []
    for seq in preds.tolist():
        chars: list[str] = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != blank:
                ch = idx_to_char.get(idx)
                if ch is not None:
                    chars.append(ch)
            prev = idx
        out.append("".join(chars))
    return out


def edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two sequences (chars or tokens)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


def cer_wer(preds: list[str], gts: list[str]) -> dict[str, float]:
    """Aggregate character and word error rates: total_edits / total_ref_len * 100."""
    char_edits = char_total = 0
    word_edits = word_total = 0
    no_err = 0
    for pred, gt in zip(preds, gts):
        ce = edit_distance(list(pred), list(gt))
        we = edit_distance(pred.split(), gt.split())
        char_edits += ce
        char_total += len(gt)
        word_edits += we
        word_total += len(gt.split())
        if pred == gt:
            no_err += 1
    return {
        "cer": 100.0 * char_edits / max(char_total, 1),
        "wer": 100.0 * word_edits / max(word_total, 1),
        "char_edits": char_edits,
        "char_total": char_total,
        "word_edits": word_edits,
        "word_total": word_total,
        "exact_match_rate": 100.0 * no_err / max(len(preds), 1),
        "n_samples": len(preds),
    }
