"""Colab'da egitilen adapter'i yerelde (CPU) yukleyip Colab sonuclariyla karsilastirir.

Colab'daki beklenen sonuclar:
  0.975  Healthy
  0.227  Bronchiectasis
  0.464  Covid19
"""

import os

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from prepare import prepare_report

BASE_MODEL_ID = "dmis-lab/biobert-v1.1"
BASE_MODEL_REVISION = "551ca18efd7f052c8dfa0b01c94c2a8e68bc5488"
DATASET_ID = "satvikt04/Chameleon-Radiology-Reports"
DATASET_REVISION = "ebf03d97a699b6ad61fc20c9a93f08b54da13f06"
ADAPTER_DIR = "models/biobert-lora-radiology"

# 1) config.json yoksa olustur: etiket isimleri (Colab'daki class_encode_column ile ayni, alfabetik sira)
if not os.path.exists(f"{ADAPTER_DIR}/config.json"):
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
    labels = sorted(set(ds["Pathology"]))
    config = AutoConfig.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={label: i for i, label in enumerate(labels)},
    )
    config.save_pretrained(ADAPTER_DIR)
    print(f"config.json olusturuldu ({len(labels)} etiket)")

# 2) BioBERT + adapter'i yukle
config = AutoConfig.from_pretrained(ADAPTER_DIR)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_ID, revision=BASE_MODEL_REVISION, config=config
)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

# 3) Colab'daki ayni 3 cumle
examples = [
    "IMPRESSION: No acute cardiopulmonary abnormality. FINDINGS: Lungs are clear. No pleural effusion.",
    "IMPRESSION: Moderate right pleural effusion with adjacent atelectasis.",
    "IMPRESSION: Filling defect in the right lower lobe pulmonary artery consistent with acute pulmonary embolism.",
]
for text in examples:
    enc = tokenizer(prepare_report(text), truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)[0]
    top = probs.argmax().item()
    print(f"{probs[top]:.3f}  {config.id2label[top]:<35} <- {text[:60]}...")
