---
type: Product Backlog Item
id: 2953
parent: 2952
title: Imagen byggs på en aktuell CUDA-bas med uppdaterade Ubuntu-paket
---

# H01 · Imagen byggs på en aktuell CUDA-bas med uppdaterade Ubuntu-paket

**Story.** Som ägare av htrflow-imagen vill jag att varje bygge utgår från
en aktuell, digest-pinnad CUDA-bas och uppdaterar Ubuntu-paketen, så att de
321 kända sårbarheterna i OS-lagret försvinner och inte återkommer vid
nästa bygge.

## Varför det är viktigt

Runtime-steget är `nvidia/cuda:12.1.0-base-ubuntu22.04` utan digest och
utan `apt-get upgrade`. Basen är från ett tidigare år: Trivy hittar 321
fixbara sårbarheter i Ubuntu-paket, 14 HIGH — gnupg-sviten (12 paket, en
HIGH var) och openssl/libssl3 — plus MEDIUM i libc, gnutls, krb5,
python3.10, libxml2, sqlite, systemd. Allt är rättat i Ubuntus egna
uppdateringar; imagen har bara aldrig hämtat dem.

## Vad som levereras

- `FROM nvidia/cuda:<aktuell 12.x-base-ubuntu22.04>@sha256:…` i både
  builder- och runtime-steget, med digest, och `apt-get upgrade -y` i
  samma `RUN` som installationen.
- gnupg-sviten och andra paket runtime inte behöver borttagna eller aldrig
  installerade (`--no-install-recommends` finns redan; verifiera vad basen
  drar in).
- Renovate-spårning av basimagens digest så att nästa CUDA-basuppdatering
  kommer som en pull request.

## Klart när

- [ ] Trivy (`--ignore-unfixed`) rapporterar 0 HIGH/CRITICAL och under 20
      MEDIUM i OS-lagret på den nya imagen.
- [ ] Dockerfilen har digest på varje `FROM`.
- [ ] htrflow-batch-wrappern bygger på den nya taggen och kör en volym
      med samma resultat som före bytet.
