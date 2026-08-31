---
type: Feature
id: 2952
parent: 2769
title: htrflow-imagen säker och uppdaterad
---

# Feature: htrflow-imagen säker och uppdaterad

## I ett stycke

`riksarkivet/htrflow` (i dag `airiksarkivet/htrflow:v0.2.6-35f48a7`, samma
digest som `latest`) är basimagen för batch-wrappern, Coder-workspaces och
externa användare. En Trivy-skanning 2026-08-31 (bara fixbara fynd) gav
**375 sårbarheter, 35 HIGH, 0 CRITICAL** — 321 i Ubuntu-paket från ett
gammalt CUDA-baslager, 54 i Python-beroenden, varav flera i verktyg som
inte borde finnas i en runtime-image. Den här featuren tar bort fynden och
ser till att de inte kommer tillbaka.

## Fastslagna egenskaper (inte stories)

- **Imagen byggs, den lappas inte.** Fixar går via Dockerfile och
  `uv.lock` i Git och en ny tagg, aldrig via `apt-get upgrade` i en
  körande container.
- **Skanningen är sanningen.** Samma Trivy-kommando som i htrflow-batch
  (`--ignore-unfixed`, CRITICAL/HIGH blockerar) körs i CI; siffrorna på
  den här sidan uppdateras därifrån.
- **Bara runtime i runtime-imagen.** Testverktyg, byggverktyg och
  pakethanterare hör hemma i builder-steget.

## Skanning 2026-08-31 (`v0.2.6-35f48a7`)

| Källa | Fixbara | HIGH | Största posterna |
|---|---|---|---|
| Ubuntu 22.04-paket | 321 | 14 | gnupg-sviten, openssl/libssl3; libc, gnutls, krb5, python3.10, xml2 (MEDIUM) |
| Python (`uv.lock`) | 54 | 21 | pillow 11.1.0 (13 HIGH), urllib3 2.3.0 (4), transformers 4.48.3 (2), Brotli, py7zr |

Fullständig JSON: `docs/audits/2026-08-31-trivy-htrflow-v0.2.6.json`.

## Stories

| ID | Azure | Story |
|---|---|---|
| H01 | #2953 | [Imagen byggs på en aktuell CUDA-bas med uppdaterade Ubuntu-paket](stories/H01-current-base-and-os-updates.md) |
| H02 | #2954 | [Sårbara Python-beroenden uppdaterade i uv.lock](stories/H02-python-deps-bumped.md) |
| H03 | #2955 | [Runtime-imagen innehåller bara det som krävs för att köra htrflow](stories/H03-runtime-only-image.md) |
| H04 | #2956 | [Varje htrflow-image skannas innan publicering och beroenden uppdateras automatiskt](stories/H04-scan-gate-and-auto-updates.md) |
| H05 | #2957 | [htrflow-imagen når SLSA Build Level 3 och konsumenter verifierar provenance](stories/H05-slsa-build-level-3.md) |
