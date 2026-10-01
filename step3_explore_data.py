import re
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_ID = "satvikt04/Chameleon-Radiology-Reports"
DATASET_REVISION = "ebf03d97a699b6ad61fc20c9a93f08b54da13f06"

MODEL_ID = "dmis-lab/biobert-v1.1"
MODEL_REVISION = "551ca18efd7f052c8dfa0b01c94c2a8e68bc5488"

ds = load_dataset(DATASET_ID, revision=DATASET_REVISION, split="train")
print(ds)

# 1) Sinif dagilimi
counts = Counter(ds["Pathology"])
print(f"\nSinif sayisi: {len(counts)}")
for label, n in counts.most_common():
    print(f"  {n:>5}  {label}")

# 2) Token uzunluklari (kesme yapmadan)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
lengths = [len(ids) for ids in tokenizer(list(ds["Report"]))["input_ids"]]
lengths.sort()

over = sum(1 for n in lengths if n > 512)
print("\nToken uzunluklari:")
print(f"  en kisa : {lengths[0]}")
print(f"  ortanca : {lengths[len(lengths) // 2]}")
print(f"  ortalama: {sum(lengths) / len(lengths):.0f}")
print(f"  en uzun : {lengths[-1]}")
print(f"  512'yi asan: {over} / {len(lengths)} ({100 * over / len(lengths):.1f}%)")

# 3) Bolum basliklari: hangi basliklar kac raporda geciyor?
reports = list(ds["Report"])
HEADER_RE = re.compile(r"\b([A-Z][A-Z /]{3,}):")
header_counts = Counter()
for r in reports:
    header_counts.update(set(HEADER_RE.findall(r)))
print("\nBolum basliklari (kac raporda var):")
for h, n in header_counts.most_common(15):
    print(f"  {n:>5}  {h}")


# 4) Sadece FINDINGS'ten sonrasini (FINDINGS + IMPRESSION ...) alirsak?
def findings_onward(text):
    m = re.search(r"FINDINGS\s*:", text, flags=re.IGNORECASE)
    return text[m.start():] if m else None


cut = [findings_onward(r) for r in reports]
missing = sum(1 for c in cut if c is None)
print(f"\nFINDINGS basligi olmayan rapor: {missing}")

cut_lengths = sorted(len(ids) for ids in tokenizer([c for c in cut if c])["input_ids"])
cut_over = sum(1 for n in cut_lengths if n > 512)
print("FINDINGS'ten itibaren token uzunluklari:")
print(f"  ortanca : {cut_lengths[len(cut_lengths) // 2]}")
print(f"  en uzun : {cut_lengths[-1]}")
print(f"  512'yi asan: {cut_over} / {len(cut_lengths)} ({100 * cut_over / len(cut_lengths):.1f}%)")

# 5) Son hal: prepare_report (IMPRESSION + FINDINGS, imza notlari atilmis)
from prepare import prepare_report

prepared = [prepare_report(r) for r in reports]
unchanged = sum(1 for p, r in zip(prepared, reports) if p == r.replace("**", "").strip())
att_left = sum(1 for p in prepared if "ATTENDING" in p.upper())
p_lengths = sorted(len(ids) for ids in tokenizer(prepared)["input_ids"])
p_over = sum(1 for n in p_lengths if n > 512)
print("\nprepare_report sonrasi:")
print(f"  basligi bulunamayan (oldugu gibi kalan): {unchanged}")
print(f"  icinde hala ATTENDING gecen: {att_left}")
print(f"  ortanca : {p_lengths[len(p_lengths) // 2]}")
print(f"  en uzun : {p_lengths[-1]}")
print(f"  512'yi asan: {p_over} / {len(p_lengths)} ({100 * p_over / len(p_lengths):.1f}%)")
