# Experiment 2 Flow

## 1. Input

The job posting dataset (`fake_job_postings.csv`) contains 17,880 job postings with two major input modalities:

- **Job text**: Combined textual fields — title, location, department, company_profile, description, requirements, and benefits — merged into a single `text` column.
- **Job screenshot**: Webpage screenshots stored as PNG images, mapped via the path `images/<fraudulent_label>/<job_id>.png`.

The target variable is `fraudulent` (0 = GENUINE, 1 = FAKE), with approximately 95.16% GENUINE and 4.84% FAKE.

---

## 2. Data Preparation

### Cleaning
- Fill missing text fields with empty strings before concatenation.
- Map each job posting to its corresponding screenshot path.
- Verify image existence; use gray placeholders for missing or corrupt images.

### Label Preparation
- Binary label: 0 = GENUINE, 1 = FAKE.
- Class mapping is explicit and consistent throughout the experiment.

### Train / Validation / Test Split
- **Three-way stratified split** using `stratify=fraudulent`.
- First split: 80% train+val, 20% test. The test set is held out immediately and never touched again.
- Second split: Within the 80%, further split into approximately 80% train and 20% validation.
- Final approximate sizes: ~70% train, ~10% validation, ~20% test.
- The validation set is **genuinely separate** from the training set. This fixes Experiment 1's bug where `val_loader` incorrectly reused `train_dataset`.

### Leakage Prevention
- TF-IDF vocabulary is fitted **only** on training data.
- BoW vocabulary is fitted **only** on training data.
- All preprocessing statistics are derived from training data only.
- Validation data is used for model selection, early stopping, and threshold tuning.
- Test data is used **only** for final evaluation after all decisions are made.

---

## 3. Text Processing

### BERT Features
```
Job Description
    → BERT Tokenizer (bert-base-uncased)
    → input_ids + attention_mask
    → BERT encoder (frozen, all parameters frozen)
    → [CLS] pooler output (768-dimensional)
    → Projection layer (768 → 256) with ReLU + BatchNorm + Dropout
    → BERT Features (256-d)
```

### TF-IDF Features
```
Job Description
    → TfidfVectorizer (fit on TRAIN data only, max_features=1000, ngram_range=(1,2))
    → 1000-dimensional sparse vector → dense
    → Projection layer (1000 → 256) with ReLU + BatchNorm + Dropout
    → TF-IDF Features (256-d)
```

### BoW Features
```
Job Description
    → CountVectorizer (fit on TRAIN data only, max_features=500, binary=True)
    → 500-dimensional sparse vector → dense
    → Projection layer (500 → 128) with ReLU + BatchNorm + Dropout
    → BoW Features (128-d)
```

---

## 4. Image Processing

### Image Preprocessing
```
Job Screenshot
    → Resize (224×224)
    → [Training only: RandomHorizontalFlip(p=0.3) + RandomRotation(5°) + ColorJitter]
    → ToTensor (scales pixels to [0,1])
    → ImageNet Normalize (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    → Image Tensor (3 × 224 × 224)
```

Training augmentation is deliberately mild to preserve screenshot layout and text readability. Validation and test preprocessing is deterministic (no augmentation).

### Pre-trained ResNet-50 (Transfer Learning)
```
Image Tensor
    → Pre-trained ResNet-50 (ImageNet weights, torchvision)
    → All convolutional backbone parameters frozen (Phase A)
    → Original FC classification layer removed (replaced with Identity)
    → Adaptive average pooling → 2048-dimensional feature vector
    → Projection layer (2048 → 256) with ReLU + BatchNorm + Dropout
    → Image Features (256-d)
```

**Transfer Learning Strategy:**

- **Phase A (default)**: Frozen backbone. Only the new projection layer and multimodal fusion components are trained. The pretrained convolutional layers act as a fixed feature extractor.
- **Phase B (optional)**: If Phase A is insufficient, unfreeze deeper ResNet layers (layer3 and layer4) and fine-tune them with a smaller learning rate (1e-5). This allows the model to adapt higher-level visual features to job-screenshot specifics.

---

## 5. Feature Fusion

```
BERT Features (256-d)     ─┐
TF-IDF Features (256-d)   ─┤
BoW Features (128-d)      ─┤→ Concatenate (896-d) → Fusion MLP → Classification Logits
ResNet-50 Features (256-d) ─┘                      (896 → 512 → 256 → 2)
                                                            ↓
                                                     GENUINE / FAKE
```

**Detailed flow:**
1. Each modality produces a feature vector of its own dimension.
2. Projection layers map each to a compatible dimension (256 or 128).
3. The four projected vectors are concatenated into a single 896-dimensional vector.
4. The fusion MLP processes this concatenated vector through two hidden layers.
5. The final output layer produces 2 logits (one per class).
6. Per-branch auxiliary classifiers provide deep supervision during training.

---

## 6. Training

- **Training set**: Used for forward pass, loss computation, and backpropagation.
- **Validation set**: Evaluated after each epoch to monitor generalization and guide early stopping.
- **Loss**: Weighted Cross-Entropy Loss with class weights computed from training data distribution. The FAKE class receives a higher weight (~10×) to compensate for the 95/5 imbalance.
- **Optimizer**: AdamW with per-module learning rates. Pretrained components use lower learning rates; new projection layers use higher rates.
- **Scheduler**: ReduceLROnPlateau that monitors validation FAKE F1 and reduces learning rate when performance plateaus.
- **Early stopping**: Monitors validation FAKE F1 with configurable patience. Stops training when FAKE F1 does not improve for `patience` consecutive epochs.
- **Checkpointing**: Saves the best model state dict based on validation FAKE F1. The checkpoint is saved to `best_experiment2_model.pth` to avoid overwriting Experiment 1 files.
- **Gradient clipping**: `max_norm=1.0` to prevent exploding gradients.
- **Mixed precision**: AMP enabled when GPU is available for faster training and lower memory usage.

---

## 7. Threshold Selection

Default classification threshold: FAKE probability ≥ 0.5 → FAKE.

However, the default threshold may not be optimal for the imbalanced fraud-detection task. A systematic threshold search is performed:

1. Collect FAKE probabilities from the model on the **validation set**.
2. Test thresholds from 0.30 to 0.70 in steps of 0.05.
3. For each threshold, compute FAKE precision, FAKE recall, and FAKE F1.
4. Select the threshold that **maximizes FAKE F1** on the validation set.

**Why validation, not test?** The test set must remain untouched during model development. Tuning the threshold on the test set would constitute data leakage — the model would indirectly learn from test-set patterns. The validation set serves as a proxy for unseen data during development.

After threshold selection, the chosen threshold is applied **once** to the held-out test set for final evaluation.

---

## 8. Final Evaluation

The test set is **completely untouched** during:
- Model training
- Early stopping decisions
- Checkpoint selection
- Threshold tuning

After all development decisions are made, the best model is loaded and evaluated on the test set **once** with the selected threshold. This produces the final, unbiased performance estimates.

---

## 9. Metrics

| Metric | Description | Importance |
|--------|-------------|------------|
| Accuracy | Overall correct predictions / total | **Secondary** — misleading under class imbalance |
| Precision (FAKE) | True FAKE / (True FAKE + False FAKE) | How many predicted fakes are actually fake |
| Recall (FAKE) | True FAKE / (True FAKE + Missed FAKE) | **Primary** — how many actual fakes are caught |
| F1 (FAKE) | Harmonic mean of FAKE precision and recall | **Primary** — balances precision and recall |
| Macro F1 | Unweighted average of per-class F1 | Shows performance across both classes equally |
| Weighted F1 | Weighted average of per-class F1 | Accounts for class imbalance |
| Confusion Matrix | TP, TN, FP, FN breakdown | Visualizes error types |
| False Negatives | Actual FAKE classified as GENUINE | **Critical** — missed fraud is the main risk |

The project's practical objective is to detect fraudulent job postings. Missing fake postings (false negatives) is more harmful than incorrectly flagging genuine postings (false positives). Therefore, **FAKE recall** and **FAKE F1** are the primary metrics, while accuracy is secondary.

---

## 10. Ablation Studies

Seven controlled experiments to determine which modalities actually contribute to fraud detection:

| Experiment | Modalities Used | Purpose |
|------------|----------------|---------|
| A | BERT only | Test semantic text understanding alone |
| B | TF-IDF only | Test statistical text features alone |
| C | BoW only | Test binary lexical features alone |
| D | ResNet-50 only | Test pretrained image features alone |
| E | BERT + TF-IDF + BoW | Test text-only multimodal (no images) |
| F | Text (BERT + TF-IDF) + ResNet-50 | Full model without BoW |
| G | Full multimodal (all four) | Complete model — baseline for comparison |

Each ablation trains a fresh model with only the specified branches and evaluates on the validation set. The purpose is to answer: Does the image branch add value beyond text features? Which text branch contributes most to fraud detection?

---

## 11. Experiment 1 vs Experiment 2

### What Changes
- **Image backbone**: Custom Multi-Kernel ResNet (trained from scratch) → Pre-trained ResNet-50 with transfer learning (ImageNet weights)
- **Validation split**: Incorrectly reused training data → Genuinely separate stratified validation set
- **Threshold analysis**: Not performed → Systematic threshold tuning on validation set
- **Monitoring metric**: Validation loss → Validation FAKE F1
- **Evaluation scope**: Basic metrics → Comprehensive metrics with FAKE-focused analysis
- **Ablation testing**: Not performed → 7 controlled ablation experiments

### What Remains Constant
- BERT branch (frozen, bert-base-uncased, 768→256 projection)
- TF-IDF branch (1000 features, ngram (1,2), 1000→256 projection)
- BoW branch (500 features, binary, 500→128 projection)
- Multimodal fusion architecture concept (projection → concatenation → MLP → classification)
- Dataset (same source, same preprocessing)
- Class mapping (0 = GENUINE, 1 = FAKE)
- Image preprocessing (ImageNet normalization, 224×224)

---

## 12. Final Research Question

> **"Does replacing the custom Multi-Kernel ResNet image branch with a pretrained ResNet-50 transfer-learning branch improve fraudulent-job detection, particularly FAKE recall and FAKE F1, while maintaining strong overall performance?"**

The answer requires comparing Experiment 2 results against Experiment 1 on the same held-out test set, with the same evaluation metrics, and with proper validation methodology.
