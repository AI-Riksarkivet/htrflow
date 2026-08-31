---
type: Product Backlog Item
id: 2954
parent: 2952
title: Sårbara Python-beroenden uppdaterade i uv.lock
---

# H02 · Sårbara Python-beroenden uppdaterade i `uv.lock`

**Story.** Som ägare av htrflow-imagen vill jag att Python-beroendena med
kända fixbara sårbarheter är höjda till rättade versioner och att htrflow
fortfarande ger samma transkriptioner, så att imagen inte bär på 21 HIGH i
bibliotek vi använder varje gång vi läser en bild.

## Varför det är viktigt

Trivy hittar 54 fixbara sårbarheter i venv:n. De som väger tyngst:
**pillow 11.1.0** (13 HIGH — bildavkodning, exakt det htrflow gör med
okända filer), **urllib3 2.3.0** (4 HIGH), **transformers 4.48.3** (2
HIGH), Brotli 1.1.0, py7zr 0.20.8, samt MEDIUM i requests, Jinja2,
filelock, idna, fonttools, python-dotenv och torch 2.6.0. Alla har
rättade versioner på PyPI.

## Vad som levereras

- `uv.lock` uppdaterad: pillow ≥ 12.3, urllib3 ≥ 2.7, requests ≥ 2.33,
  Brotli ≥ 1.2, py7zr ≥ 1.1.3, Jinja2 ≥ 3.1.6, filelock, idna, fonttools,
  python-dotenv till rättade versioner.
- transformers höjd inom 4.x (≥ 4.53) — 5.x är känt inkompatibelt med
  htrflows modeller — med testsviten och ett referensdokument som
  regressionsskydd.
- torch: höjd till 2.8+ *om* CUDA-basen (H01) tillåter det; annars
  dokumenterat undantag med CVE-id och skäl.
- Övre gränser i `pyproject.toml` där en major-höjning skulle bryta
  (transformers `<5`), så att Renovate inte föreslår den.

## Klart när

- [ ] Trivy rapporterar 0 HIGH i Python-paket; kvarvarande MEDIUM är
      listade med skäl i changelogen.
- [ ] Testsviten grön; en referenssida transkriberad före/efter ger samma
      text (eller en dokumenterad förbättring).
- [ ] `uv.lock` innehåller inget paket med kända fixbara HIGH enligt
      `uv pip audit`/Trivy i CI.
