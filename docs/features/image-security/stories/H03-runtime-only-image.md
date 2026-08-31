---
type: Product Backlog Item
id: 2955
parent: 2952
title: Runtime-imagen innehåller bara det som krävs för att köra htrflow
---

# H03 · Runtime-imagen innehåller bara det som krävs för att köra htrflow

**Story.** Som ägare av htrflow-imagen vill jag att den publicerade
imagen bara innehåller htrflow och dess runtime-beroenden — inte
testverktyg, pakethanterare eller byggverktyg — så att angreppsytan och
skanningsbruset krymper och imagen blir mindre att dra på varje GPU-nod.

## Varför det är viktigt

Venv:n i den publicerade imagen innehåller `pytest`, `virtualenv` och en
gammal `uv 0.5.31` (tre MEDIUM), vilket betyder att `uv sync` körs med
dev-gruppen inkluderad och att verktyg följer med från builder-steget.
Ingenting av det används när htrflow kör en pipeline. Varje onödigt paket
är en framtida CVE att förklara och megabyte att hämta.

## Vad som levereras

- `uv sync --frozen --no-dev` i builder-steget; dev-beroenden
  (`pytest`, lint, docs) flyttade till en `dev`-grupp i `pyproject.toml`.
- Runtime-steget kopierar bara `.venv` och `src`; ingen `uv`, ingen
  `pip`, inga kompilatorer.
- Imagen kör som en icke-root-användare med fast uid, så att konsumenter
  (batch-wrappern kör redan som 1000) inte behöver lägga till det själva.

## Klart när

- [ ] `pip list`/`uv pip list` i den publicerade imagen visar varken
      pytest, virtualenv eller uv.
- [ ] Imagen är minst 10 % mindre än `v0.2.6-35f48a7` (komprimerad).
- [ ] `docker run … id` visar en icke-root-användare; htrflow-batch
      wrappern startar utan ändring.
