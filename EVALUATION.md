# Test Set Evaluation Report

This document contains the comprehensive evaluation metrics for all components of the Arabic Bank Check Amount Extraction system on the unseen 600-image test set (`Test Set/CheckImages-Test`). All results reflect the end-to-end "pipeline" approach (where later parts depend on the actual outputs of earlier parts, rather than oracle ground-truths).

---

## Part A: Region of Interest Detection (YOLOv8)
Evaluates the capability of the YOLO model to correctly place bounding boxes over the Legal and Courtesy amount regions.

- **Total Images Evaluated:** 600
- **Overall Mean IoU (Intersection over Union):** 0.5978

**Legal Amount Metrics:**
- **meanIoU**: 0.7254
- **Accuracy @ IoU ≥ 0.50**: 96.67%
- **Accuracy @ IoU ≥ 0.75**: 44.33%
- **Accuracy @ IoU ≥ 0.90**: 5.83%
- **Missed Detections:** 1 / 600

**Courtesy Amount Metrics:**
- **meanIoU**: 0.4703
- **Accuracy @ IoU ≥ 0.50**: 43.00%
- **Accuracy @ IoU ≥ 0.75**: 6.33%
- **Accuracy @ IoU ≥ 0.90**: 0.83%
- **Missed Detections:** 3 / 600

---

## Part B: Courtesy Amount Recognition (CRNN Pipeline)
Evaluates the CRNN model's ability to extract digits from the courtesy bounding boxes predicted by Part A. 

- **Total Crops Evaluated:** 597 (3 missed by YOLO)
- **Digit Accuracy:** 98.11%
- **Digit Error Rate:** 1.88%
- **Raw Character Error Rate:** 2.08%

**Sample-Level Error Distribution:**
- **0 Errors (Perfect Extraction):** 92.96%
- **1 Error:** 6.03%
- **2+ Errors:** 1.00%

---

## Part C: Legal Amount Recognition (TrOCR-PAW Pipeline)
Evaluates the VisionEncoderDecoder (TrOCR) model's ability to extract Piece-of-Arabic-Word (PAW) sequences from the legal bounding boxes predicted by Part A.

- **Total Crops Evaluated:** 599 (1 missed by YOLO)
- **PAW Word Error Rate (WER):** 40.49%
- **PAW Character Error Rate (CER - Joined):** 18.03%
- **Raw Character Error Rate (CER - Concat):** 12.22%
- **Exact Match Rate:** 2.50%

---

## Part D: End-to-End Verification Pipeline
Evaluates the overall system coherence by parsing the textual output of Part C into integers and mathematically comparing it to the digit output of Part B.

- **Total Images Evaluated:** 600

**Matching Outcomes:**
- **MATCH** (Amounts match mathematically): **405** (67.50%)
- **MISMATCH** (Amounts contradict each other): **194** (32.33%)
- **UNPARSED** (Legal amount text was incomprehensible): **1** (0.17%)

> [!NOTE] 
> Because this is a blind test set, "Blame Attribution" (diagnosing whether Part B or Part C was at fault for a mismatch) is omitted, as it requires raw ground truth plaintext equivalents which are absent from this test set structure.
