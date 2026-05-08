"""CTC decoders + metrics for PAW-token predictions.

  greedy_paw_decode  : argmax over PAW logits, collapse repeats, drop blank.
  beam_paw_decode    : prefix-beam-search over PAW logits.
  paw_metrics        : PAW-WER + PAW-CER (joined-string char distance) + raw-CER.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from PartC.decode import edit_distance
from PartC.paw_tokenizer import PAWTokenizer, join_paws


def greedy_paw_decode(log_probs: torch.Tensor, tokenizer: PAWTokenizer,
                      blank: int = 0) -> list[list[str]]:
    """log_probs: (T, B, V). Returns list of PAW lists, length B."""
    preds = log_probs.argmax(dim=2).transpose(0, 1)  # (B, T)
    out: list[list[str]] = []
    for seq in preds.tolist():
        toks: list[str] = []
        prev = -1
        for idx in seq:
            if idx != prev and idx != blank:
                tok = tokenizer.id_to_paw.get(int(idx))
                if tok is not None and tok != "<blank>":
                    toks.append(tok)
            prev = idx
        out.append(toks)
    return out


def beam_paw_decode(log_probs: torch.Tensor, tokenizer: PAWTokenizer,
                    beam_size: int = 10, blank: int = 0) -> list[list[str]]:
    """Prefix-beam-search CTC over PAW logits.

    Standard variant from Hannun et al. 2014 (no LM rescoring here; that's
    optional in evaluate_paw via --lm).
    """
    log_probs = log_probs.detach().cpu()
    T, B, V = log_probs.shape
    results: list[list[str]] = []
    for b in range(B):
        # beams: dict prefix(tuple of ids) -> (logp_blank_end, logp_nonblank_end)
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
        for t in range(T):
            new_beams: dict[tuple[int, ...], tuple[float, float]] = defaultdict(
                lambda: (-math.inf, -math.inf)
            )
            log_p = log_probs[t, b].numpy()  # (V,)
            for prefix, (pb, pnb) in beams.items():
                # 1) emit blank: extend both b and nb endings with blank.
                cur_b, cur_nb = new_beams[prefix]
                merged_b = _logsumexp(cur_b, _logsumexp(pb, pnb) + log_p[blank])
                new_beams[prefix] = (merged_b, cur_nb)

                # 2) emit a non-blank token c.
                # Iterate over top-K candidates only to keep cost down.
                topk_idx = log_p.argsort()[-beam_size - 1 :][::-1]
                for c in topk_idx:
                    c = int(c)
                    if c == blank:
                        continue
                    log_c = float(log_p[c])
                    if prefix and prefix[-1] == c:
                        # repeated token: only allowed via blank-ended prefix.
                        cur_b, cur_nb = new_beams[prefix]
                        new_beams[prefix] = (cur_b, _logsumexp(cur_nb, pnb + log_c))
                        new_prefix = prefix + (c,)
                        cb, cnb = new_beams[new_prefix]
                        new_beams[new_prefix] = (cb, _logsumexp(cnb, pb + log_c))
                    else:
                        new_prefix = prefix + (c,)
                        cb, cnb = new_beams[new_prefix]
                        new_beams[new_prefix] = (
                            cb,
                            _logsumexp(cnb, _logsumexp(pb, pnb) + log_c),
                        )
            # prune
            scored = sorted(
                new_beams.items(),
                key=lambda kv: -_logsumexp(kv[1][0], kv[1][1]),
            )[:beam_size]
            beams = dict(scored)

        # pick best
        best = max(beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]))
        prefix = best[0]
        toks = [tokenizer.id_to_paw.get(int(i), "<unk>") for i in prefix]
        toks = [t for t in toks if t != "<blank>"]
        results.append(toks)
    return results


def _logsumexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def paw_metrics(preds: list[list[str]], gts: list[list[str]]) -> dict[str, float]:
    """Aggregate WER on PAW tokens + CER on the joined-string + CER on the
    no-space concat (a proxy for raw)."""
    paw_edits = paw_total = 0
    join_char_edits = join_char_total = 0
    raw_char_edits = raw_char_total = 0
    no_err = 0
    for p_paws, g_paws in zip(preds, gts):
        paw_edits += edit_distance(p_paws, g_paws)
        paw_total += len(g_paws)

        p_join = join_paws(p_paws)
        g_join = join_paws(g_paws)
        join_char_edits += edit_distance(list(p_join), list(g_join))
        join_char_total += len(g_join)

        p_raw = "".join(p_paws)
        g_raw = "".join(g_paws)
        raw_char_edits += edit_distance(list(p_raw), list(g_raw))
        raw_char_total += len(g_raw)

        if p_paws == g_paws:
            no_err += 1
    return {
        "paw_wer": 100.0 * paw_edits / max(paw_total, 1),
        "paw_cer_joined": 100.0 * join_char_edits / max(join_char_total, 1),
        "raw_cer_concat": 100.0 * raw_char_edits / max(raw_char_total, 1),
        "paw_edits": paw_edits,
        "paw_total": paw_total,
        "exact_match_rate": 100.0 * no_err / max(len(preds), 1),
        "n_samples": len(preds),
    }
