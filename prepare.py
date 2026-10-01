"""Rapor metnini modele vermeden once hazirlar.

Egitimde (Colab) ve serviste (FastAPI) AYNI fonksiyon kullanilmali,
yoksa model serviste egitimde gormedigi bicimde metin alir.
"""

import re

# Rapordaki bolum basliklari. "**FINDINGS:**" gibi markdown kalin yaziyi da yakalar.
SECTION_RE = re.compile(
    r"\**\s*(CLINICAL INFORMATION|PROCEDURE|COMPARISON|FINDINGS|IMPRESSION"
    r"|RECOMMENDATIONS?|ATTENDING (?:PHYSICIAN|RADIOLOGIST) AGREEMENT)\s*\**\s*:\s*\**",
    flags=re.IGNORECASE,
)

# Imza notunun baslik disindaki yazimlari:
# "[ATTENDING PHYSICIAN AGREEMENT]: ..." satiri veya "... by the attending radiologist." cumlesi
ATTENDING_RE = re.compile(
    r"\[ATTENDING[^\]]*\][^\n]*|[^.\n]*\battending (?:radiologist|physician)\b[^.\n]*\.?",
    flags=re.IGNORECASE,
)


def split_sections(text: str) -> dict[str, str]:
    """Raporu {"FINDINGS": "...", "IMPRESSION": "...", ...} sozlugune boler."""
    matches = list(SECTION_RE.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[name] = text[m.end():end].strip()
    return sections


def prepare_report(text: str) -> str:
    """IMPRESSION + FINDINGS dondurur; diger bolumleri (imza notu vb.) atar.

    IMPRESSION once gelir: 512 token sinirinda kesilme olursa
    teshis bolumu degil, Findings'in sonu kesilir.
    Baslik bulunamazsa (or. kullanicinin yazdigi kisa metin) metni oldugu gibi dondurur.
    """
    sections = split_sections(text)
    parts = []
    if sections.get("IMPRESSION"):
        parts.append("IMPRESSION: " + sections["IMPRESSION"])
    if sections.get("FINDINGS"):
        parts.append("FINDINGS: " + sections["FINDINGS"])

    result = "\n\n".join(parts) if parts else text
    # Imza notlari sinifla iliskili (kisayol ogrenme riski), teshisle ilgisiz: at.
    result = ATTENDING_RE.sub("", result)
    # Markdown kalin yazi isaretleri (**) model icin anlamsiz, sadece token harcar.
    return result.replace("**", "").strip()
