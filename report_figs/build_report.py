"""Build ~10-page academic DOCX report for the Fake ML project (Exp1 + Exp2).
All numbers are taken from executed notebook outputs; nothing is invented."""
from docx import Document
from docx.shared import Pt, Mm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import os

FIG = r"C:\DEV\FAKE ML\report_figs"
OUT = r"C:\DEV\FAKE ML\Fake_Job_Fraud_Detection_Experiments_Report.docx"
NAVY = RGBColor(0x1F, 0x3B, 0x63)

doc = Document()

# ---- page setup: A4, compact academic margins ----
for s in doc.sections:
    s.page_width, s.page_height = Mm(210), Mm(297)
    s.top_margin = s.bottom_margin = Mm(20)
    s.left_margin = s.right_margin = Mm(20)

# ---- base styles ----
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.paragraph_format.line_spacing = 1.5
normal.paragraph_format.space_after = Pt(4)
normal.paragraph_format.space_before = Pt(0)
for i in (1, 2, 3):
    h = doc.styles[f"Heading {i}"]
    h.font.name = "Calibri"
    h.font.color.rgb = NAVY
    h.font.bold = True
    h.paragraph_format.space_before = Pt(8)
    h.paragraph_format.space_after = Pt(3)
    h.paragraph_format.line_spacing = 1.15
doc.styles["Heading 1"].font.size = Pt(13)
doc.styles["Heading 2"].font.size = Pt(11)
doc.styles["Heading 3"].font.size = Pt(10.5)


def add_field(par, instr):
    r1 = par.add_run(); f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r1._r.append(f1)
    r2 = par.add_run(); i = OxmlElement("w:instrText"); i.set(qn("xml:space"), "preserve"); i.text = instr; r2._r.append(i)
    r3 = par.add_run(); f3 = OxmlElement("w:fldChar"); f3.set(qn("w:fldCharType"), "end"); r3._r.append(f3)


# ---- footer: Page X ----
fp = doc.sections[0].footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = fp.add_run(); r.font.size = Pt(9); r.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
add_field(fp, "PAGE")


def h1(t): return doc.add_heading(t, level=1)
def h2(t): return doc.add_heading(t, level=2)
def para(t, bold_prefix=None, italic=False):
    p = doc.add_paragraph()
    if bold_prefix:
        r = p.add_run(bold_prefix); r.bold = True
    r = p.add_run(t)
    if italic: r.italic = True
    return p
def bullets(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")
def table(headers, rows, widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, hh in enumerate(headers):
        c = t.cell(0, j)
        c.text = ""
        r = c.paragraphs[0].add_run(hh); r.bold = True; r.font.size = Pt(9)
        shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:fill"), "D9E2F3")
        c._tc.get_or_add_tcPr().append(shd)
    for i, row in enumerate(rows, 1):
        for j, v in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            r = c.paragraphs[0].add_run(str(v)); r.font.size = Pt(9)
    for row in t.rows:
        row.height_rule = None
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t
def figure(path, caption, width_mm=138):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4); p.paragraph_format.space_after = Pt(1)
    p.add_run().add_picture(path, width=Mm(width_mm))
    c = doc.add_paragraph(); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(6)
    r = c.add_run(caption); r.italic = True; r.font.size = Pt(9)


# ================= TITLE PAGE =================
for _ in range(3): doc.add_paragraph()
tp = doc.add_paragraph(); tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = tp.add_run("Detecting Fraudulent Job Postings\nwith Multimodal Machine Learning"); r.bold = True
r.font.size = Pt(20); r.font.color.rgb = NAVY
sp = doc.add_paragraph(); sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sp.add_run("Experiment 1: Classical Baselines and Custom Multi-Kernel ResNet  •  Experiment 2: Pretrained ResNet-50 Transfer Learning")
r.font.size = Pt(11); r.italic = True
mp = doc.add_paragraph(); mp.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = mp.add_run("Project report  •  Source of truth: executed notebooks noteX.ipynb and Experiment_2_Multimodal_ResNet50.ipynb  •  September 2026")
r.font.size = Pt(10)
box = doc.add_paragraph(); box.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = box.add_run("Headline result — Exp1 deep model: FAKE F1 0.747, recall 0.786 (55 FP / 37 FN)  •  Exp2: FAKE F1 0.786, precision 0.951 (6 FP / 57 FN). "
                "Transfer learning cut false alarms 9× but missed 20 more frauds.")
r.font.size = Pt(10); r.bold = True
doc.add_page_break()

# ================= TOC =================
h1("Contents")
tcp = doc.add_paragraph()
add_field(tcp, 'TOC \\o "1-2" \\h \\z \\u')
doc.add_paragraph("Right-click the table of contents in Word and choose “Update Field” to populate page numbers.",
                  style="Intense Quote")

# ================= 1 =================
h1("1.  Introduction and Problem Statement")
para("Online job portals receive thousands of postings, a small fraction of which are fraudulent: phishing lures, "
     "advance-fee scams, or fake recruiters harvesting personal data. Manual moderation does not scale, so automated "
     "classification of postings as genuine (0) versus fake (1) is a practical necessity.")
para("This project treats fake-job detection as a supervised binary classification problem over two modalities: the "
     "posting text (title, description, requirements and related fields) and a screenshot of the posting web page. "
     "The machine-learning objective is to maximise fraud detection — catching fake postings (recall) without flooding "
     "moderators with false alarms (precision) — under severe class imbalance (≈95% genuine / 5% fake), where raw "
     "accuracy is misleading. Two controlled experiments were conducted on the same corpus: Experiment 1 establishes "
     "classical baselines and a from-scratch multimodal network; Experiment 2 fixes a validation bug from Experiment 1 "
     "and replaces the custom image backbone with a pretrained ResNet-50.")

# ================= 2 =================
h1("2.  Dataset and Overall Methodology")
h2("2.1  Dataset")
para("Both experiments use fake_job_postings.csv (17,880 postings, 18 original columns) plus per-posting screenshot "
     "PNGs mapped as images/<fraudulent>/<job_id>.png, with a grey placeholder substituted for missing or corrupt images. "
     "The target fraudulent is binary (0 = genuine, 1 = fake). Table 1 summarises the data; Figure 1 shows the imbalance.")
table(["Item", "Detail"],
      [["Source file", "fake_job_postings.csv + screenshot PNGs"],
       ["Samples", "17,880 postings"],
       ["Target", "fraudulent: 17,014 genuine (95.16%) vs 866 fake (4.84%)"],
       ["Text fields (merged)", "title, location, department, company_profile, description, requirements, benefits"],
       ["Structured fields", "telecommuting, has_company_logo, has_questions, employment_type, required_experience, required_education, industry, function, salary_range, job_id"],
       ["Images", "Web-page screenshots, 224×224 input, ImageNet normalisation"]])
figure(f"{FIG}/fig1_class_dist.png", "Figure 1 — Target class distribution: 95.16% genuine vs 4.84% fake.", 120)
h2("2.2  Data characteristics")
bullets([
    "Missingness is heavy in optional fields: salary_range (~15,000 missing), department (~11,500), required_education, benefits and experience are all incomplete; description is missing only once. Missing text is filled with empty strings before concatenation.",
    "Fraudulent postings are shorter (mean ≈286 vs ≈383 words) and disproportionately lack company logos, with unusual employment-type patterns — a signal the lexical branches exploit.",
    "No data leakage in feature fitting: TF-IDF and BoW vocabularies are fitted on training text only, then applied unchanged to validation/test.",
])
h2("2.3  Splits and training discipline")
para("Experiment 1 uses a stratified 80/20 split (seed 42): 14,304 train / 3,576 test, both preserving the 4.84% fake ratio. "
     "Experiment 2 uses a three-way stratified split from the same seed: 11,443 train (64.0%) / 2,861 validation (16.0%) / "
     "3,576 test (20.0%), each at ≈4.84–4.86% fake. The first-stage 80/20 operation is identical, so both experiments share "
     "the same held-out test set (3,403 genuine + 173 fake) and their test metrics are directly comparable. Class imbalance "
     "is handled by class-weighted loss (≈20:1 fake weight) rather than resampling in the deep models; the classical "
     "baselines additionally test SMOTE/ADASYN oversampling of the training set only. Evaluation always uses the original "
     "imbalanced test set, with FAKE recall and FAKE F1 as primary metrics and accuracy treated as secondary.")

# ================= 3 =================
h1("3.  Experiment 1 — Baselines and Custom Multimodal Network")
h2("3.1  Objective")
para("Establish (a) classical TF-IDF + logistic-regression baselines under three imbalance strategies, and (b) a "
     "from-scratch four-branch multimodal network, evaluated on the held-out test set.")
h2("3.2  Baselines: features, models, results")
para("TF-IDF (max 1,000 features, 1–2-grams) fitted on training text feeds L2 logistic regression under three regimes: "
     "Normal (imbalanced), SMOTE (k=5, balanced to 13,611/13,611 in TF-IDF space), ADASYN (same balancing, denser near "
     "hard minority regions). Figure 2 previews the outcome; Table 2 gives exact test figures.")
figure(f"{FIG}/fig2_baselines.png", "Figure 2 — FAKE-class precision/recall/F1: classical baselines vs the Experiment 1 deep model.", 138)
table(["Method", "Acc.", "FAKE prec.", "FAKE rec.", "FAKE F1", "ROC-AUC", "FP", "FN"],
      [["LR Normal", "0.9673", "0.9375", "0.3468", "0.5063", "0.9655", "4", "113"],
       ["LR SMOTE", "0.9550", "0.5205", "0.8786", "0.6538", "0.9751", "140", "21"],
       ["LR ADASYN", "0.9365", "0.4258", "0.8960", "0.5773", "0.9711", "209", "18"]])
para("Verdict on baselines: Normal hides fraud (113 of 173 fakes missed despite 96.7% accuracy); SMOTE is the best "
     "classical trade-off (F1 0.654, AUC 0.975); ADASYN gains 2 recall points but nearly doubles false alarms (209 FP) — "
     "its hard-region synthesis adds noise.")
h2("3.3  Deep model: representation and training")
bullets([
    "Branches (all frozen or from-scratch as noted): BERT-base-uncased frozen, [CLS] 768-d → 256; TF-IDF 1,000-d → 256; binary BoW 500-d → 128; custom MultiKernelResNet (stem 7×7 conv + maxpool, four stages of parallel 3×3/5×5/7×7 blocks with residual shortcuts, GAP → 512-d) trained from scratch → 256. Each projection uses ReLU + BatchNorm + dropout 0.4.",
    "Fusion: concatenated 896-d → MLP 896→512→256→2, with per-branch auxiliary classifiers (deep supervision) and learnable modality weights α, β, γ, δ.",
    "Loss/optimiser: class weights [0.525, 10.32] in an adaptive multi-task loss (fusion CE + learned-weighted branch CEs); AdamW with per-module rates (3e-4 fresh heads, 1e-4 BERT/image projections, 1e-5 CNN backbone, 2e-4 fusion, 1e-3 modality weights), weight decay 0.01, gradient clipping (norm 1.0), AMP, batch size 8, 5 epochs (≈19–22 min/epoch on an RTX 3050 6 GB laptop GPU). Checkpoint on validation loss; fixed 0.5 threshold; no early stopping, no threshold tuning.",
])
h2("3.4  Results")
table(["Metric", "Value"],
      [["Accuracy", "0.9743"], ["FAKE precision", "0.7120"], ["FAKE recall", "0.7861"],
       ["FAKE F1", "0.7473"], ["ROC-AUC", "0.9656"], ["Macro / weighted F1", "0.87 / 0.97"],
       ["Confusion matrix", "[[3348, 55], [37, 136]]  (55 FP, 37 FN)"]])
figure(f"{FIG}/fig3_cm_exp1.png", "Figure 3 — Experiment 1 test confusion matrix at threshold 0.50.", 105)
para("Key findings: the multimodal net beats every baseline (F1 0.747 vs 0.654, AUC 0.966) and balances both error types "
     "(55 FP / 37 FN). Caveat: the validation loader was constructed from the training dataset (val_loader = "
     "DataLoader(train_dataset, …)), so validation loss/model selection were optimistic — the test metrics above remain "
     "valid because the test set was never touched, but checkpoint choice was made on leaked validation signal.")

# ================= 4 =================
h1("4.  Experiment 2 — Pretrained ResNet-50 Transfer Learning")
h2("4.1  Objective and differences from Experiment 1")
para("Test whether replacing the from-scratch MultiKernelResNet with an ImageNet-pretrained ResNet-50 (transfer learning) "
     "improves fraud detection, while fixing Experiment 1’s validation bug. Everything else is held comparable: identical "
     "BERT/TF-IDF/BoW branches and projections, identical 896→512→256→2 fusion, same 0.4 dropout and auxiliary heads — "
     "minus the learnable modality weights. The full decision log (15 items) is recorded in decision.md; the substantive "
     "changes are: (1) genuine train/val/test split; (2) frozen ResNet-50 backbone (2048-d → 256 projection; optional "
     "Phase-B unfreezing of layer3/4 documented but run under frozen Phase A); (3) plain weighted cross-entropy instead of "
     "the adaptive multi-task loss; (4) checkpointing and early stopping on validation FAKE F1; (5) systematic threshold "
     "search on validation; (6) FAKE-focused metrics (macro/weighted F1, FP/FN counts).")
h2("4.2  Training")
para("Weighted cross-entropy (balanced weights, fake ≈10×); AdamW per-module (1e-5 frozen backbones, 1e-4/3e-4 projections, "
     "2e-4 fusion), weight decay 0.01; ReduceLROnPlateau on validation FAKE F1 (factor 0.5, patience 3); gradient clipping, "
     "AMP, batch 8, max 20 epochs with early stopping (patience 5). Training stopped at epoch 11; the best checkpoint "
     "(epoch 6, validation FAKE F1 0.8065) was kept. Mild screenshot augmentation only (flip p=0.3, ±5° rotation, colour "
     "jitter); validation/test preprocessing deterministic. Figure 7 shows the history: train loss falls throughout while "
     "validation FAKE F1 peaks at epoch 6 and early stopping fires after five stagnant epochs.")
figure(f"{FIG}/fig7_training.png", "Figure 7 — Experiment 2 training history: loss (left) and validation FAKE F1 peaking at epoch 6 (right).", 138)
h2("4.3  Threshold selection (validation only)")
para("Thresholds 0.30–0.70 (step 0.05) were scored on the validation set (n=2,861) maximising FAKE F1; 0.40 was selected "
     "(validation FAKE F1 0.8080, 10 FP / 38 FN). Tuning never touched the test set.")
table(["Thr.", "0.30", "0.35", "0.40★", "0.45", "0.50", "0.60", "0.70"],
      [["Val FAKE F1", "0.8016", "0.8048", "0.8080", "0.8080", "0.8065", "0.8016", "0.7984"],
       ["Val FAKE prec.", "0.8938", "0.9018", "0.9099", "0.9099", "0.9174", "0.9167", "0.9327"],
       ["Val FAKE rec.", "0.7266", "0.7266", "0.7266", "0.7266", "0.7194", "0.7122", "0.6978"]])
figure(f"{FIG}/fig6_threshold.png", "Figure 6 — Validation threshold sweep: flat F1 plateau 0.40–0.55; 0.40 selected (highest recall on the plateau).", 132)
h2("4.4  Final test results (threshold 0.40, applied once)")
table(["Metric", "Value"],
      [["Accuracy", "0.9824"], ["FAKE precision", "0.9508"], ["FAKE recall", "0.6705"],
       ["FAKE F1", "0.7864"], ["Macro / weighted F1", "0.8886 / 0.9809"],
       ["Confusion matrix", "TN 3397, FP 6, FN 57, TP 116"],
       ["ROC-AUC", "Not computed in the notebook — not available"]])
figure(f"{FIG}/fig4_cm_exp2.png", "Figure 4 — Experiment 2 test confusion matrix at the selected threshold 0.40.", 105)
para("Key findings: near-perfect precision (95.1%; only 6 false alarms) with moderate recall (67.1%; 57 of 173 fakes missed). "
     "The validation→test FAKE F1 gap is small (0.808 → 0.786), indicating honest generalisation. The seven designed "
     "ablation configurations (BERT/TF-IDF/BoW/ResNet-only, text-only, text+ResNet, full) are defined in the notebook but "
     "were not executed — no ablation results exist, so modality contributions remain unmeasured.")

# ================= 5 =================
h1("5.  Experiment 1 vs Experiment 2")
table(["Aspect", "Experiment 1", "Experiment 2"],
      [["Test set", "3,576 (3,403 G / 173 F)", "Identical 3,576 (same seed/split)"],
       ["Train / val", "14,304 train; val = train (bug)", "11,443 train / 2,861 genuine val"],
       ["Image branch", "Custom MultiKernelResNet, from scratch (512-d)", "ResNet-50 ImageNet-pretrained, frozen (2048-d)"],
       ["Loss / selection", "Adaptive multi-task loss; ckpt on val loss", "Weighted CE; ckpt + early stop on val FAKE F1"],
       ["Threshold", "Fixed 0.50", "Tuned on val → 0.40"],
       ["Accuracy", "0.9743", "0.9824 (+0.0081)"],
       ["FAKE precision", "0.7120", "0.9508 (+0.2388)"],
       ["FAKE recall", "0.7861", "0.6705 (−0.1156)"],
       ["FAKE F1", "0.7473", "0.7864 (+0.0391)"],
       ["Macro / weighted F1", "0.87 / 0.97", "0.8886 / 0.9809"],
       ["ROC-AUC", "0.9656", "Not available"],
       ["FP / FN", "55 / 37", "6 / 57"]])
figure(f"{FIG}/fig5_exp1_exp2.png", "Figure 5 — Head-to-head test metrics (left) and error counts (right).", 138)
para("Technical analysis: Experiment 2 is not a Pareto improvement — it trades recall for precision. The pretrained "
     "backbone plus FAKE-F1-driven selection and a lower threshold operating point raised precision by 24 points and cut "
     "false alarms 55→6, lifting FAKE F1 by 3.9 points, macro F1 and accuracy with it. But FAKE recall fell 11.6 points "
     "(37→57 missed frauds). Because both share the same test set, the comparison is apples-to-apples; the caveat is "
     "attribution — the gain bundles four changes (backbone, honest validation/early stopping, FAKE-F1 checkpointing, "
     "threshold tuning), so the ResNet-50 alone cannot be credited. The improvement is meaningful where moderator workload "
     "dominates (9× fewer false alarms for +3.9 F1 points) but is a regression where missing fraud dominates: 20 extra "
     "missed fakes is a 54% increase in false negatives. Overall the better experiment depends on cost asymmetry; on the "
     "project’s headline metric (FAKE F1), Experiment 2 wins.")

# ================= 6 =================
h1("6.  Overall Findings and Discussion")
bullets([
    "Best approach by FAKE F1: Experiment 2 (0.786) ahead of Experiment 1 deep (0.747) and SMOTE-LR (0.654). Best fraud-catcher by recall: SMOTE-LR (0.879) and ADASYN-LR (0.896) — at 140–209 false alarms — then Exp1 deep (0.786); Exp2 trails at 0.671.",
    "Precision/recall operating points differ starkly: Exp1 deep 0.712/0.786 (balanced), Exp2 0.951/0.671 (conservative). Exp2 flags almost only sure fraud; Exp1 catches ~12 points more fraud at ~9× the false-alarm rate.",
    "False negatives are the critical error: a missed fake posting exposes job seekers to scams, while a false positive merely costs a manual review. Exp1 misses 37 fakes, Exp2 misses 57 — Exp2’s headline win comes entirely from eliminating false positives (55→6), not from catching more fraud.",
    "Class imbalance dominates design: accuracy (0.97–0.98) is uninformative — the Normal baseline “scores” 0.967 while missing two-thirds of fraud. Weighted loss, resampling, and FAKE-first metrics are what make the task learnable.",
    "Feature observations: short text, missing logos and lexical cues carry fraud signal (baselines reach 0.88 recall on TF-IDF alone); the image branch’s marginal value is unproven — ablations were never run, and Exp2’s gain cannot be isolated to ResNet-50.",
    "Unexpected result: transfer learning plus rigorous selection improved precision far more than recall — the opposite of the naive “better backbone catches more fraud” expectation; the conservative operating point suggests pretrained screenshot features are discriminative but the tuned threshold favours certainty.",
])
para("Practical reading: deploy Exp2 where review capacity is scarce (6 false alarms per 3,576 postings); deploy Exp1 "
     "(or the SMOTE baseline) where missing fraud is costlier, accepting higher review load. A cost-weighted threshold, "
     "rather than either fixed point, is the natural next operating choice.")

# ================= 7 =================
h1("7.  Limitations and Future Improvements")
h2("7.1  Limitations (grounded in the implementation)")
bullets([
    "Small minority: only 866 fakes total (554 train in Exp2); deep multimodal training on a few hundred positives risks overfitting and noisy recall estimates (±1 fake ≈ 0.6 recall points on test).",
    "Exp1 validation leak: checkpoint selection on training-data-as-validation; mitigated in Exp2 but Exp1’s reported operating point carries selection bias.",
    "Attribution gap: Exp2 changes backbone, validation, checkpoint metric and threshold simultaneously — no isolation of the ResNet-50 effect; ablations A–G unexecuted.",
    "Missing metrics: no ROC-AUC or PR curve for Exp2, so ranking quality across thresholds cannot be compared with Exp1’s 0.9656.",
    "Limited tuning: single seed, single split, fixed 0.5 (Exp1) / coarse 0.05-grid (Exp2) thresholds, no cross-validation, Phase-B fine-tuning untried, frozen BERT throughout.",
    "Feature gaps: structured signals (logo presence, employment type, text length) enter only implicitly via text/screenshots; placeholder grey images for missing screenshots add noise.",
])
h2("7.2  Future improvements")
bullets([
    "Execute the seven ablations plus PR/ROC curves to isolate the image branch’s value and compare ranking quality.",
    "Phase-B ResNet fine-tuning (layer3/4, lr 1e-5), BERT unfreezing/LoRA, and a cost-weighted threshold reflecting FN≫FP harm.",
    "K-fold or repeated-seed evaluation with confidence intervals; calibration of fake probabilities.",
    "Explicit structured features (logo, length, employment type) and modern text encoders; external or synthetic fraud augmentation beyond SMOTE.",
])

# ================= 8 =================
h1("8.  Conclusion")
para("Two experiments investigated multimodal fake-job detection on 17,880 postings at a 95/5 imbalance. Experiment 1 "
     "showed classical TF-IDF logistic regression tops out at F1 0.654 (SMOTE) while a from-scratch four-branch network "
     "(frozen BERT + custom MultiKernelResNet + TF-IDF + BoW) reached accuracy 0.9743, FAKE F1 0.7473 and recall 0.7861 "
     "with 55 FP / 37 FN — despite a validation-split bug. Experiment 2 fixed validation, adopted a frozen pretrained "
     "ResNet-50, FAKE-F1 checkpointing/early stopping and validation-tuned threshold 0.40, achieving accuracy 0.9824 and "
     "FAKE F1 0.7864 with only 6 false alarms but 57 missed frauds. On the headline FAKE-F1 metric Experiment 2 performs "
     "better (+3.9 points, 9× fewer false alarms); on fraud recall Experiment 1 is better (−11.6 points, 20 fewer misses). "
     "The most important finding is the trade-off itself: rigorous transfer learning made the model far more precise but "
     "more conservative. Recommended direction: run the pending ablations, fine-tune Phase B, and choose the operating "
     "threshold from explicit false-negative/false-positive costs rather than F1 alone.")

doc.save(OUT)
print("saved:", OUT, os.path.getsize(OUT), "bytes")
