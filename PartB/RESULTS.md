# Part B — Courtesy Amount Recognition Results

## Setup
- **Dataset**: 1431 train / 177 val / 177 test (Part A image-stem splits).
  Tokenized labels (1799 entries in `Dataset/CourtesyAmounts/`) used as canonical;
  raw labels (1497 entries) ignored for training due to coverage gap.
- **Vocab**: 12 symbols `{0-9, '.', '/'}` + blank = 13 (train-only). 0% OOV in val/test.
- **Crops**: GT bboxes (class 1) cropped from `Dataset/CheckImages/*.tif` with margin=2.
- **Architecture**: ResNet-18 CRNN (PartC/model.py), input H=96, W/4 downsample.
- **Training**: AdamW lr=1e-3 cosine+warmup, batch=32, AMP, light aug (±2° rotation,
  ±2 px translate, ±10% brightness/contrast, σ=2 noise), 100 epochs patience=15.
- **Loss**: CTC over 13-class vocab. Decode: greedy.
- **Best epoch**: 32 (early-stopped at 47).

## Test results

The four metrics required by the project PDF (computed on digit-only strings —
`.` and `/` stripped from BOTH pred and GT before scoring):

| Eval mode | Digit accuracy | % no-error | % one-error | % two-or-more |
|---|---|---|---|---|
| Val (oracle, greedy)            | **97.58%** | 91.53% | 6.78%  | 1.69% |
| Test (oracle, greedy)           | **95.37%** | 85.88% | 10.17% | 3.95% |
| Test (oracle, beam=10)          | 95.24%     | 86.44% | 9.60%  | 3.95% |
| Test (pipeline, YOLO + CRNN)    | **91.36%** | 79.66% | 12.43% | 7.91% |

Plan targets (≥ 95% digit accuracy, ≥ 80% no-error on test oracle): **MET**.

YOLO end-to-end miss rate: 11 / 1800 checks (0.6%) had no class-1 detection.
All 177 test images were detected.

## Files

- Training log: `PartB/runs/crnn_b1/log.csv`, `PartB/runs/crnn_b1.log`
- Best weights: `PartB/runs/crnn_b1/weights/best.pt`
  (absolute: `C:\Users\qxawe\NLP_Project\.claude\worktrees\elated-meninsky-959dc4\PartB\runs\crnn_b1\weights\best.pt`)
- Predictions file (PDF format `<filename> <digits>`):
  `PartB/runs/crnn_b1/predictions_courtesy.txt` (1800 lines, 11 with empty digits)
- Per-sample JSON outputs: `val.json`, `test_oracle.json`, `test_beam.json`,
  `test_pipeline.json` under `PartB/runs/crnn_b1/`.

## Notes for the report

- Beam=10 didn't beat greedy → encoder features, not decoding, are at the limit.
  Skipped Exp 3 (TrOCR-digit) per plan stop-condition.
- 4 pp drop oracle → pipeline is from YOLO bbox jitter, not detection failures.
  Tighter NMS or slightly larger YOLO margin could narrow this gap.
- `/` only appears 2× in train (a fraction-bar in mixed-format checks).
  Underrepresented at training time, but the metric strips it anyway, so no impact.
- Digit `0` dominates the train distribution (2246 occurrences vs 288 for `9`).
  Confusion matrix candidates for the report: `1`↔`7`, `5`↔`6`, `0`↔`o-shaped` rotations.
