# Radiology Report Classifier API (BioBERT + LoRA)

> ⚠️ **Not for clinical use.** Educational / portfolio project. Trained on **synthetic** reports; performance on
> real clinical reports has not been validated. Do not use it for diagnosis, triage or any patient decision.

A FastAPI service that reads an English chest CT radiology report and returns one of 18 classes
(17 pathologies + Healthy) with a probability. The model is
[BioBERT](https://huggingface.co/dmis-lab/biobert-v1.1) fine-tuned with **LoRA** (PEFT), trained on a free
Google Colab T4 GPU in ~10 minutes.

- **Model on Hugging Face:** [busum/biobert-lora-chameleon-radiology](https://huggingface.co/busum/biobert-lora-chameleon-radiology)
- **Training notebook:** [train_lora.ipynb](train_lora.ipynb)

```
report text ──▶ prepare_report() ──▶ BioBERT tokenizer ──▶ BioBERT + LoRA adapter ──▶ label + probability
                (IMPRESSION + FINDINGS,                   (only 0.28% of the
                 signature notes removed)                  parameters trained)
```

## Why these choices

| Decision | Reason |
|---|---|
| **BioBERT** instead of a larger model (e.g. BioMedLM) | Biomedical pre-training, 108M parameters — fits free Colab for training and a CPU for serving |
| **LoRA** instead of full fine-tuning | 308,754 trainable parameters (0.28%); the adapter is 1.2 MB instead of ~400 MB |
| **Synthetic dataset** (Chameleon) | Clear terms, labels included, no real patient data. Credentialed datasets such as MIMIC were deliberately not used |
| **IMPRESSION + FINDINGS only** | 51% of raw reports exceeded BioBERT's 512-token limit and the impression (the most diagnostic part) sits at the end. Keeping impression first + findings brings it to 1%, and truncation never removes the impression |
| **Signature notes removed** | "ATTENDING PHYSICIAN AGREEMENT" appeared in 30%–66% of reports depending on the class — a shortcut the model could learn instead of the findings |
| **Same `prepare.py` in training and serving** | Avoids training–serving skew: no error is raised, predictions just silently get worse |
| **Pinned versions and revisions** | Libraries with `==`, base model, adapter and dataset with Hub commit hashes |

## Results

Held-out test set (1,000 reports, stratified 80/10/10 split):

| Accuracy | Macro-F1 |
|---|---|
| **0.995** | **0.991** |

Errors concentrate in clinically overlapping classes (lowest F1: Adenocarcinoma 0.947, Metastatic Nodule 0.951).
Adenocarcinoma is itself a subtype of non-small cell lung cancer, so these labels are not mutually exclusive.

## Limitations — the most important section

The test score is high because train and test reports come from the same generator with near-identical phrasing.
To check what the model really learned, I wrote my own short reports (not taken from the dataset):

| Input I wrote | Prediction | Correct? |
|---|---|---|
| "IMPRESSION: No acute cardiopulmonary abnormality. FINDINGS: Lungs are clear. No pleural effusion." | Healthy 0.975 | ✅ |
| "Large right pleural effusion." | Covid19 0.368 | ❌ |
| Same finding written report-style (`FINDINGS: … IMPRESSION: Large right pleural effusion.`) | Pleural Effusion 0.284 | ✅ but unsure |
| "IMPRESSION: Filling defect in the right lower lobe pulmonary artery consistent with acute pulmonary embolism." | Covid19 0.465 | ❌ |

**Takeaway:** the model learned the dataset's template more than the medical concepts. Changing only the format
changed the answer. Out-of-distribution inputs get low confidence, so the API flags predictions below 0.5 with
`"low_confidence": true`. Real, multi-site reports would be needed before drawing any clinical conclusion.

Other limits: English only, single label per report, chest CT reports only, max 512 tokens.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and pinned adapter revision |
| POST | `/predict` | Report text → label, probability, top-k, flags |

Request:

```json
{"text": "IMPRESSION: No acute cardiopulmonary abnormality. FINDINGS: Lungs are clear.", "top_k": 3}
```

Response:

```json
{
  "label": "Healthy",
  "probability": 0.9573,
  "low_confidence": false,
  "top": [
    {"label": "Healthy", "probability": 0.9573},
    {"label": "Covid19", "probability": 0.0143},
    {"label": "Bronchitis", "probability": 0.0081}
  ],
  "truncated": false,
  "inference_ms": 109.9,
  "disclaimer": "Not for clinical use. Educational project trained on synthetic reports."
}
```

`truncated` is `true` when the prepared text is longer than 512 tokens and its end was cut.

## Run locally (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port 8001
```

Open http://localhost:8001/docs. The first start downloads BioBERT (~430 MB) and the adapter (~2 MB).

```powershell
curl.exe -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d '{\"text\": \"IMPRESSION: Large right pleural effusion.\"}'
```

## Run with Docker

```bash
docker build -t radiology-report-api .
docker run -p 8001:8001 radiology-report-api
```

The model is downloaded at **build** time and baked into the image, so the container starts without internet
access (`HF_HUB_OFFLINE=1`). In GitHub Codespaces, add port 8001 manually in the **PORTS** tab.

## Project structure

```
app/model.py           loads BioBERT + adapter from the Hub (pinned revisions), runs prediction
app/main.py            FastAPI app: lifespan model loading, /health, /predict
prepare.py             report preprocessing shared by training and serving
train_lora.ipynb       Colab notebook: data split, LoRA training, evaluation, saving
step1..step4_*.py      learning steps: pipeline API, tokenizer, data exploration, local adapter check
Dockerfile             CPU image, non-root user, model baked in
```

Development extras (data exploration): `pip install -r requirements-dev.txt`.

**Known issue (Colab):** if `get_peft_model` fails with *"incompatible version of torchao"*, run
`!pip uninstall -y torchao` and restart the session. Colab ships an old torchao that PEFT rejects; this project
does not use it.

## Data, license and citation

- **Code:** educational project.
- **Model adapter:** CC BY-NC 4.0 (non-commercial), see the model card.
- **Training data:** [Chameleon Radiology Reports](https://huggingface.co/datasets/satvikt04/Chameleon-Radiology-Reports) —
  synthetic reports; research, educational and non-commercial use only; **redistribution not allowed**.
  The dataset is **not** included in this repository.
- **Base model:** BioBERT (dmis-lab), Apache 2.0.

> Tripathi S., Enwerem D., Alkhulaifat D., Sukumaran R., Chambers C., Levic D., Daye D., Cook T.
> *Chameleon Dataset: A Large Multi-Pathology Synthetic Chest CT Radiology Reports Dataset* (2025).
> SSRN: https://ssrn.com/abstract=5386019 · doi:10.2139/ssrn.5386019

> Lee J. et al. *BioBERT: a pre-trained biomedical language representation model for biomedical text mining.*
> Bioinformatics 36(4):1234–1240 (2020).
