# Experiment 2 Decision Log

This file tracks every important architectural and methodological decision across both experiments. Experiment 2 results are left blank until the notebook is manually executed.

---

## Decision Table

| # | Component | Experiment 1 | Experiment 2 | Reason for Change |
|---|-----------|---------------|---------------|-------------------|
| 1 | **Image backbone** | Custom Multi-Kernel ResNet (parallel 3×3, 5×5, 7×7 convolutions, 512-d output) | Pre-trained ResNet-50 from torchvision (2048-d output) | Standard pretrained backbone provides stronger, more generalizable image features via transfer learning. Enables fair comparison with established architectures. |
| 2 | **Image pretraining** | Trained from scratch on job screenshots | ImageNet-pretrained (torchvision `ResNet50_Weights.DEFAULT`) | From-scratch training on a small, domain-specific dataset risks overfitting. ImageNet features capture universal visual patterns (edges, textures, layouts) that transfer well to screenshot analysis. |
| 3 | **Transfer learning** | None (full training from scratch) | Phase A: Frozen backbone, only new layers trained. Phase B (optional): Unfreeze layer3+layer4 with smaller LR | Controlled transfer learning prevents catastrophic forgetting of pretrained features while allowing adaptation to job-screenshot specifics. Two phases provide flexibility. |
| 4 | **Image preprocessing** | Resize(224×224) → ToTensor → ImageNet Normalize; training augmentation (flip/rotation/color jitter) | Resize(224×224) → ToTensor → ImageNet Normalize; controlled mild training augmentation | Augmentation is kept mild to avoid destroying text/layout information in job screenshots. Validation/test preprocessing is deterministic. |
| 5 | **Validation split** | **BUG**: `val_loader = DataLoader(train_dataset, ...)` — validation reused the training dataset | **FIX**: Separate stratified split → genuine validation dataset with its own DataLoader | A true validation set is required for honest model selection, early stopping, and threshold tuning. Reusing training data produces misleadingly optimistic validation metrics. |
| 6 | **Class imbalance strategy** | Weighted CrossEntropyLoss (balanced weights) + adaptive multi-task loss with learnable modality weights (α, β, γ, δ) | Weighted CrossEntropyLoss (balanced weights) as primary method. Focal loss documented as alternative | Weighted CE is simpler, more interpretable, and sufficient as a starting point. The adaptive multi-task loss added complexity without clearly established benefit. Alternatives documented for future exploration. |
| 7 | **Fusion architecture** | Per-branch projection → concatenation (896-d) → fusion MLP (896→512→256→2); per-branch auxiliary classifiers; learnable modality weights | Per-branch projection → concatenation (896-d) → fusion MLP (896→512→256→2); per-branch auxiliary classifiers | Conceptually comparable to Experiment 1 for fair comparison. Removed learnable modality weights to reduce complexity. |
| 8 | **BERT** | bert-base-uncased, frozen, [CLS] pooler output (768-d) → projection (768→256) | bert-base-uncased, frozen, [CLS] pooler output (768-d) → projection (768→256) | Kept identical for fair comparison. Freezing avoids overfitting on the small training set. |
| 9 | **TF-IDF** | TfidfVectorizer(max_features=1000, ngram_range=(1,2)) → projection (1000→256) | TfidfVectorizer(max_features=1000, ngram_range=(1,2)) → projection (1000→256) | Kept identical for fair comparison. Fitted only on training data. |
| 10 | **BoW** | CountVectorizer(max_features=500, binary=True) → projection (500→128) | CountVectorizer(max_features=500, binary=True) → projection (500→128) | Kept identical for fair comparison. Experiment 1 showed BoW responded to suspicious wording when other branches did not. |
| 11 | **Threshold selection** | Default threshold 0.5 (no tuning) | Systematic threshold search on validation set (0.30–0.70, step 0.05); threshold maximizing FAKE F1 selected | Default threshold may not be optimal for the imbalanced fraud-detection task. Tuning on validation data (not test) finds a better operating point. |
| 12 | **Evaluation metrics** | Accuracy, Precision, Recall, F1, ROC-AUC, classification report, confusion matrix | All of the above PLUS: Macro F1, Weighted F1, explicit FAKE recall, FAKE F1, false positive count, false negative count | The project's practical objective is fraud detection. Missing fake postings is more critical than overall accuracy. Explicit FAKE-focused metrics make this visible. |
| 13 | **Checkpointing** | Saved best model based on validation loss | Saved best model based on validation FAKE F1 | Monitoring FAKE F1 directly optimizes for the research goal (improving fraud detection), rather than overall loss which is dominated by the majority GENUINE class. |
| 14 | **Ablation testing** | Not performed | Seven controlled ablation experiments (BERT-only, TF-IDF-only, BoW-only, ResNet-50-only, text-only, text+image, full model) | Ablation studies determine which modalities actually contribute to fraud detection and reveal whether the image branch adds value beyond text features. |
| 15 | **Early stopping** | Not implemented (fixed 5 epochs) | Implemented, monitoring validation FAKE F1 with configurable patience | Prevents overfitting and stops training when fraud-detection performance plateaus. |

---

## Experiment 1 Known Issues Addressed

| Issue | Experiment 1 | Experiment 2 Fix |
|-------|---------------|-------------------|
| Validation data leak | `val_loader` created from `train_dataset` — validation metrics were training metrics in disguise | Separate stratified validation split with its own DataLoader |
| No threshold tuning | Fixed threshold 0.5 with no exploration | Systematic threshold search on validation set, selecting for FAKE F1 |
| Custom image backbone from scratch | Multi-Kernel ResNet trained on ~14k screenshots | Pretrained ResNet-50 with ImageNet transfer learning |
| Limited evaluation metrics | Basic accuracy/precision/recall/F1 | Comprehensive metrics with explicit FAKE-focused analysis (FAKE recall, FAKE F1, FP, FN counts) |
| No ablation studies | Single full model only | 7 controlled ablation experiments to isolate modality contributions |
| No early stopping | Fixed 5 epochs regardless of convergence | Early stopping on validation FAKE F1 with patience |
| Checkpoint based on loss | Best model selected by validation loss (dominated by majority class) | Best model selected by validation FAKE F1 (directly optimizes fraud detection) |

---

## Experiment 1 Results (Reference)

These are the Experiment 1 results to compare against once Experiment 2 is executed:

| Metric | Experiment 1 Value |
|--------|-------------------|
| Accuracy | 0.9743 |
| Precision | 0.7120 |
| FAKE Recall | 0.7861 |
| FAKE F1 | 0.7473 |
| Confusion Matrix | [[3348, 55], [37, 136]] |
| False Positives | 55 |
| False Negatives | 37 |

---

## Experiment 2 Results

> **PLACEHOLDER — Do not fill until the notebook is manually executed.**

| Metric | Experiment 2 Value |
|--------|-------------------|
| Accuracy | — |
| Precision | — |
| FAKE Recall | — |
| FAKE F1 | — |
| Macro F1 | — |
| Weighted F1 | — |
| False Positives | — |
| False Negatives | — |
| Selected Threshold | — |
