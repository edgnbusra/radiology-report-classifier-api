# ---------------------------------------------------------------
# Radyoloji Raporu Sınıflandırma API - Docker imajı (CPU)
# Build:  docker build -t radiology-report-api .
# Run:    docker run -p 8001:8001 radiology-report-api
# ---------------------------------------------------------------

# 1) Temel imaj: küçük bir Debian Linux + Python 3.11 (yerelde kullandığımız sürüm)
FROM python:3.11-slim

# Python'un .pyc dosyası yazmasını ve çıktıyı tamponlamasını kapat (loglar anında görünsün)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 2) Paketler. requirements.txt koddan ÖNCE kopyalanır: kod değişince Docker bu ağır
#    katmanı önbellekten kullanır, paketleri yeniden kurmaz.
#    torch'u ayrıca CPU index'inden kuruyoruz; Linux'ta PyPI'deki varsayılan torch
#    CUDA'lı sürümdür ve imajı birkaç GB büyütür.
COPY requirements.txt .
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# 3) Root olmayan kullanıcı (güvenlik: container ele geçirilse bile yetki sınırlı kalır)
RUN useradd --create-home appuser
USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

# 4) Uygulama kodu. prepare.py eğitimde de kullanılan metin hazırlama fonksiyonu.
COPY --chown=appuser:appuser prepare.py ./
COPY --chown=appuser:appuser app/ ./app/

# 5) BioBERT (~430 MB) + LoRA adapter'ı BUILD sırasında Hub'dan indirip imaja göm:
#    container açılışta internete ihtiyaç duymaz. Revision'lar app/model.py'de sabit.
RUN python -c "from app.model import ReportClassifier; ReportClassifier().load()"
# Bundan sonra Hugging Face'e bağlanmaya çalışma, sadece önbellekteki modeli kullan.
ENV HF_HUB_OFFLINE=1

EXPOSE 8001

# Docker'ın container sağlığını takip etmesi için (/health endpoint'i)
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

# 0.0.0.0: container dışından (senin tarayıcından) gelen isteklere de açık ol
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
