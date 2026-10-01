"""
"Garson": HTTP isteklerini alır, modele iletir, cevabı döndürür.
Çalıştırma:  .venv/Scripts/uvicorn app.main:app --port 8001
Test arayüzü: http://localhost:8001/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import (
    ADAPTER_ID,
    ADAPTER_REVISION,
    BASE_MODEL_ID,
    LOW_CONFIDENCE_THRESHOLD,
    ReportClassifier,
)

DISCLAIMER = "Not for clinical use. Educational project trained on synthetic reports."

classifier = ReportClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sunucu açılırken BİR KEZ çalışır: model her istekte değil, burada yüklenir
    classifier.load()
    yield
    # Sunucu kapanırken buraya düşer (temizlenecek kaynak yok)


app = FastAPI(
    title="Radyoloji Raporu Sınıflandırma API",
    description=(
        f"`{BASE_MODEL_ID}` + LoRA adapter `{ADAPTER_ID}` ile göğüs BT raporlarını 18 sınıfa ayırır.\n\n"
        "**Uyarı:** Yalnızca eğitim amaçlıdır, klinik kullanım için değildir. "
        "Sentetik raporlarla eğitildi; gerçek raporlarda başarısı doğrulanmadı."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=20_000,
        description="İngilizce radyoloji raporu metni",
        examples=["IMPRESSION: No acute cardiopulmonary abnormality. FINDINGS: Lungs are clear."],
    )
    top_k: int = Field(3, ge=1, le=18, description="Döndürülecek en olası etiket sayısı")


class LabelProbability(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    label: str
    probability: float
    low_confidence: bool = Field(description=f"Olasılık {LOW_CONFIDENCE_THRESHOLD} altındaysa true")
    top: list[LabelProbability]
    truncated: bool = Field(description="Hazırlanmış metin 512 token'dan uzunsa true (sonu kesildi)")
    inference_ms: float
    disclaimer: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier.is_loaded,
        "adapter": ADAPTER_ID,
        "adapter_revision": ADAPTER_REVISION,
    }


# Not: Endpoint'ler "async def" değil düz "def". Model CPU'yu meşgul eden, senkron bir iştir;
# FastAPI düz "def" fonksiyonları ayrı bir iş parçacığında çalıştırır, böylece sunucu kilitlenmez.
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Metin boş olamaz.")
    pred = classifier.predict(req.text, top_k=req.top_k)
    return PredictResponse(
        label=pred.label,
        probability=round(pred.probability, 4),
        low_confidence=pred.low_confidence,
        top=[LabelProbability(label=l, probability=round(p, 4)) for l, p in pred.top],
        truncated=pred.truncated,
        inference_ms=round(pred.inference_ms, 1),
        disclaimer=DISCLAIMER,
    )
