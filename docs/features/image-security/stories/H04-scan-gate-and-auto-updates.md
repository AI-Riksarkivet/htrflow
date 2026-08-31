---
type: Product Backlog Item
id: 2956
parent: 2952
title: Varje htrflow-image skannas innan publicering och beroenden uppdateras automatiskt
---

# H04 · Varje htrflow-image skannas innan publicering och beroenden uppdateras automatiskt

**Story.** Som ägare av htrflow-imagen vill jag att ingen image publiceras
med kända fixbara HIGH/CRITICAL-sårbarheter, och att nya rättningar i bas
och beroenden kommer som pull requests av sig själva, så att H01–H03 inte
är en engångsstädning utan ett tillstånd som håller.

## Varför det är viktigt

`build-and-push-docker.yml` körs för hand, hämtar `uv:latest` opinnat,
skannar ingenting och signerar ingenting. Ingen Renovate/Dependabot bevakar
`uv.lock` eller basimagen. Det är så en image från förra året fortfarande
är `latest`. htrflow-batch har redan mönstret: Trivy blockerar på
CRITICAL/HIGH, cosign signerar, Renovate bevakar images och lockfiler.

## Vad som levereras

- Trivy-steg i bygget (samma flaggor som htrflow-batch:
  `--ignore-unfixed`, CRITICAL/HIGH ger fel) som stoppar publiceringen;
  SARIF till GitHub Security.
- Publicering på release-tagg (inte manuellt), digest-pinnad `uv`,
  cosign-signering och provenance som i htrflow-batch `publish.yml`; mål
  `riksarkivet/htrflow` (B61 i htrflow-batch).
- Renovate konfigurerad för `uv.lock`, Dockerfile-baser (digest) och
  GitHub Actions; en schemalagd skanning av senaste publicerade image
  varje vecka som öppnar ett issue vid nya HIGH.

## Klart när

- [ ] Ett bygge med en känd HIGH-sårbarhet stoppas i CI; samma bygge
      passerar efter en beroendehöjning.
- [ ] Nästa release publiceras av taggen, signerad; `cosign verify` går
      igenom.
- [ ] Renovate har öppnat minst en PR mot `uv.lock` och en mot basimagen.
