"""
"Mutfak": Modeli yükleme, metin hazırlama, tahmin.
FastAPI'den tamamen bağımsızdır; step4_local_check.py ile aynı yükleme mantığını kullanır.
"""
import time
from dataclasses import dataclass

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from prepare import prepare_report

BASE_MODEL_ID = "dmis-lab/biobert-v1.1"
BASE_MODEL_REVISION = "551ca18efd7f052c8dfa0b01c94c2a8e68bc5488"
# Colab'da eğittiğimiz LoRA adapter'ı. Hub'daki depo değişse bile bu sürümü kullanırız.
ADAPTER_ID = "busum/biobert-lora-chameleon-radiology"
ADAPTER_REVISION = "b48ae529e5e9320c2b0104598f66c0b50b80f761"

MAX_TOKENS = 512
# Kendi yazdığımız cümlelerde yanlış tahminlerin güveni 0.23 ve 0.46 idi; altını güvenilmez sayıyoruz.
LOW_CONFIDENCE_THRESHOLD = 0.5


@dataclass
class Prediction:
    label: str
    probability: float
    top: list[tuple[str, float]]   # en olası k etiket, olasılığa göre azalan
    n_tokens: int                  # hazırlanmış metnin token sayısı (kesmeden önce)
    inference_ms: float

    @property
    def low_confidence(self) -> bool:
        return self.probability < LOW_CONFIDENCE_THRESHOLD

    @property
    def truncated(self) -> bool:
        return self.n_tokens > MAX_TOKENS


class ReportClassifier:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.id2label: dict[int, str] = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        # İlk çalıştırmada Hugging Face'ten indirir (~430 MB BioBERT + ~2 MB adapter), sonra önbellekten okur
        config = AutoConfig.from_pretrained(ADAPTER_ID, revision=ADAPTER_REVISION)  # 18 etiket ismi burada
        self.tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, revision=ADAPTER_REVISION)
        # "classifier MISSING" uyarısı normal: bir sonraki satırda adapter eğitilmiş classifier'ı takıyor
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL_ID, revision=BASE_MODEL_REVISION, config=config
        )
        self.model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_REVISION)
        self.model.eval()
        self.id2label = config.id2label

    def predict(self, text: str, top_k: int = 3) -> Prediction:
        # Eğitimdeki metin hazırlamanın AYNISI (training-serving skew olmasın)
        prepared = prepare_report(text)
        n_tokens = len(self.tokenizer(prepared)["input_ids"])
        enc = self.tokenizer(prepared, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")

        t0 = time.perf_counter()
        with torch.inference_mode():                    # gradyan yok -> daha hızlı, daha az bellek
            probs = self.model(**enc).logits.softmax(-1)[0]
        inference_ms = (time.perf_counter() - t0) * 1000

        values, indices = probs.topk(top_k)
        top = [(self.id2label[i], float(p)) for p, i in zip(values.tolist(), indices.tolist())]
        return Prediction(
            label=top[0][0],
            probability=top[0][1],
            top=top,
            n_tokens=n_tokens,
            inference_ms=inference_ms,
        )
