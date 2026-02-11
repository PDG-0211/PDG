# PDG (Parameter Description Generation)

PDG aims to **generate natural-language descriptions for function/method parameters** from code context.

This repo includes two baselines:
- **CodeBERT (RoBERTa) encoder + Transformer decoder** (`code/cb/`)
- **CodeT5/T5-based models** (`code/ct5/`)

---

## Project layout

```text
code/
  cb/     # CodeBERT training & models
  ct5/    # CodeT5/T5 training, models, and metrics
```

---

## Dataset

Dataset link (provided by the original repository):
- Google Drive: https://drive.google.com/file/d/1zCrmVSPixZeDiCkYWnmJ5XVomkYdVQmQ/view?usp=drive_link

Expected files:
```text
data/<dataset_name>/{train,valid,test}.jsonl
```

Each line is a Python-dict-like string and read with `eval(...)` in the training scripts.
---

## Installation

Python 3.8+ is recommended.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Training

### CodeT5/T5
```bash
cd code/ct5
bash script/train_base.sh
```

### CodeBERT
```bash
cd code/cb
bash script/train_base.sh
```

You may need to replace `from_pretrained(r"")` placeholders with a Hugging Face model name or a local checkpoint path.

---

## Outputs & metrics

Test results are written under your `output_dir` and evaluated with BLEU / ROUGE-L / METEOR / Exact Match (see `code/ct5/metrics/`).