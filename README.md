# i23-6004 — Assignment 2: Neural NLP Pipeline
**Course:** CS-4063 Natural Language Processing
**University:** FAST National University of Computer & Emerging Sciences
**Framework:** PyTorch (from scratch — no HuggingFace, Gensim, or pretrained models)

---

## Folder Structure

```
i23-6004_Assignment2_DS-X/
│
├── i23-6004_Assignment2DS_B_.ipynb     # Main notebook (all cells executed)
├── report.pdf                           # Report (Times New Roman, 12pt, 2–3 pages)
│
├── embeddings/
│   ├── tfidf_matrix.npy                 # TF-IDF weighted term-document matrix
│   ├── ppmi_matrix.npy                  # PPMI weighted word-word co-occurrence matrix
│   ├── embeddings_w2v.npy               # Final averaged Skip-gram embeddings (V+U)/2
│   └── word2idx.json                    # Vocabulary: word → integer index (9,594 entries)
│
├── models/
│   ├── bilstm_pos.pt                    # Best BiLSTM POS model (fine-tuned, no CRF)
│   ├── bilstm_ner.pt                    # Best BiLSTM NER model (fine-tuned, no CRF)
│   └── transformer_cls.pt               # Transformer classifier state dict
│
└── data/
    ├── pos_train.conll                  # POS annotations — train split (CoNLL format)
    ├── pos_test.conll                   # POS annotations — test split (CoNLL format)
    ├── ner_train.conll                  # NER annotations — train split (CoNLL format)
    └── ner_test.conll                   # NER annotations — test split (CoNLL format)
```

---

## Requirements

```
torch
numpy
scikit-learn
matplotlib
seqeval
tqdm
seaborn
```

Install all dependencies:
```bash
pip install torch numpy scikit-learn matplotlib seqeval tqdm seaborn
```

---

## Input Files Required

Place these in the same directory as the notebook before running:

| File | Used In | Purpose |
|------|---------|---------|
| `cleaned.txt` | Parts 1, 2, 3 | Primary training corpus (11,546 sentences) |
| `raw.txt` | Part 1 | Ablation baseline (unprocessed corpus) |
| `metadata.json` | Part 3 | Topic labels for 215 articles |

---

## How to Run

Open and run `i23-6004_Assignment2DS_B_.ipynb` top to bottom. All cells are pre-executed with outputs visible.

### Part 1 — Word Embeddings
- **Cells 1–2:** Builds TF-IDF matrix, PPMI matrix, t-SNE visualisation, and cosine similarity neighbours. Saves `tfidf_matrix.npy` and `ppmi_matrix.npy`.
- **Cell 4:** Trains Skip-gram Word2Vec for 5 epochs. Saves `embeddings_w2v.npy`.

### Part 2 — Sequence Labeling
- **Cell 7:** Annotates 500 sentences with POS and NER tags; splits 70/15/15 into `train.json`, `val.json`, `test.json`.
- **Cell 10:** Trains BiLSTM under 5 configurations (POS/NER × frozen/fine-tuned × CRF). Saves `.pt` model files.
- **Cell 13:** Full evaluation — classification reports, confusion matrices, false positive/negative analysis.

### Part 3 — Transformer Classification
- **Cell 15:** Loads metadata, encodes articles to 256-token sequences, splits 70/15/15.
- **Cell 17:** Trains Transformer encoder (4 layers, d=128, h=4) for 20 epochs with AdamW + cosine LR schedule.
- **Cell 20–22:** Test evaluation, confusion matrix, attention heatmaps for 3 correctly classified articles.

### File Export (run after training)
- **Cell 24:** Saves `word2idx.json`
- **Cell 25:** Converts `train.json`/`test.json` → CoNLL format files
- **Cell 26:** Saves `transformer_cls.pt`
- **Cell 27:** Organises all outputs into the submission folder structure

---

## Key Results Summary

### Part 1 — Word Embeddings

| Condition | Description | Observation |
|-----------|-------------|-------------|
| C1 — PPMI | Co-occurrence vectors | Sparse; struggles on rare tokens |
| C2 — W2V raw.txt | Skip-gram on raw corpus | Noise degrades quality |
| C3 — W2V cleaned.txt | Skip-gram on clean corpus | Best semantic coherence ✓ |
| C4 — W2V d=200 | C3 with doubled dimension | Marginal gain; higher cost |

Skip-gram training loss: **3.2651 → 2.8495** over 5 epochs (6,487 batches/epoch).

### Part 2 — POS Tagging

| Mode | Token Accuracy | Macro-F1 | Best Val-F1 |
|------|---------------|----------|-------------|
| Frozen embeddings | 68% | 0.512 | 0.457 |
| Fine-tuned embeddings | **89%** | **0.745** | **0.701** |

Most confused pairs (fine-tuned): ADJ→NOUN (78), NOUN→ADV (33), NOUN→UNK (31).

### Part 2 — NER

| Configuration | Micro-F1 | Note |
|---------------|----------|------|
| Frozen, no CRF | 0.07 | Best LOC recall: 0.00 entity-level |
| Fine-tuned, no CRF | **0.10** | Best overall configuration |
| Fine-tuned, CRF | 0.00 | CRF diverged (loss went negative by epoch 3) |

NER was severely constrained by extreme class imbalance (O-tag = 96.41% of all tokens).

### Part 3 — Transformer Classification

| Metric | Value |
|--------|-------|
| Test Accuracy | **52.78%** |
| Macro-F1 | **0.2452** |
| Best Val Accuracy | 58.06% (epoch 15–20) |

International (F1: 0.76) and Politics (F1: 0.46) were the only learnable classes. Economy, Health & Society, and Sports scored F1 = 0.00 due to insufficient training examples (<20 each).

---

## Design Decisions

**Why fine-tuned > frozen for POS?**
Frozen embeddings were trained on general Urdu text; fine-tuning adapts them to POS-relevant contextual patterns, gaining +23.3 macro-F1 points.

**Why did the CRF collapse?**
The CRF transition matrix diverged numerically under 96.4% O-tag imbalance on a 348-sentence training set. Training loss became negative by epoch 3. The softmax model with fine-tuned embeddings (val F1: 0.327) was therefore selected as `bilstm_ner.pt`.

**Why does the Transformer underperform?**
Only 148 training articles with severe class skew (International = 44.6%). The Transformer's self-attention requires substantial data to learn meaningful patterns; the BiLSTM's sequential inductive bias is better suited to this scale.

**Why is `bilstm_pos.pt` = `pos_False_False.pt`?**
`False_False` refers to `Freeze=False, CRF=False` — i.e., fine-tuned embeddings with a linear classifier head — which achieved the best POS performance (Macro-F1: 0.745).

**Why is `bilstm_ner.pt` = `ner_False_False.pt`?**
`Freeze=False, CRF=False` achieved the best NER val-F1 (0.327). The CRF variant (`ner_False_True.pt`) collapsed and scored 0.00.

---

## CoNLL File Format

Each `.conll` file contains one token per line with its label, separated by a tab. Sentences are separated by a blank line:

```
مریم    NOUN
نواز    NOUN
نے      POST
ہدایت   NOUN
...

حکومت   NOUN
نے      POST
...
```

---

## Notebook Environment

- **Python:** 3.x
- **Device used for training:** CPU
- **W2V training time:** ~2.5 min/epoch (6,487 batches × 512 batch size)
- **Transformer training time:** ~9 sec/epoch (5 batches × 32 batch size)
- **BiLSTM training:** 15 epochs per configuration, early stopping patience = 5
