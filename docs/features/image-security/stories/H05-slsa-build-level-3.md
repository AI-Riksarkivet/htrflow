---
type: Product Backlog Item
id: 2957
parent: 2952
title: htrflow-imagen når SLSA Build Level 3 och konsumenter verifierar provenance
---

# H05 · htrflow-imagen når SLSA Build Level 3 och konsumenter verifierar provenance

**Story.** Som ägare av htrflow-imagen vill jag att varje publicerad image
har provenance på SLSA Build Level 3 — skapad av byggplattformen, inte av
vårt eget jobb, och därför inte förfalskningsbar av den som kan ändra
workflowen — och att de som kör imagen (batch-wrappern, dev-klustret)
verifierar den, så att "det här är htrflow, byggd från vår källkod, av vår
CI" är något man kontrollerar, inte antar.

## Varför det är viktigt

En signatur (H04) säger att *någon med vår CI-identitet* signerade
digesten. Level 3-provenance säger dessutom *från vilket repo, vilken
commit och vilken workflow* imagen byggdes, genererad i ett steg som
byggjobbet självt inte kan påverka. htrflow-batch har samma mål (B14,
#2867) för sina tre images; basimagen de bygger på måste hålla samma nivå,
annars är kedjan bara så stark som sin svagaste länk. Riksarkivets
Kyverno-policy för "bara våra images" kan då kräva attestationen, inte
bara signaturen.

## Vad som levereras

- Bygget flyttat till ett återanvändbart, härdat workflow som ger Level
  3-provenance: GitHubs artifact attestations via reusable workflow eller
  `slsa-framework/slsa-github-generator` (container-generatorn), enligt
  samma val som B14 så att htrflow och htrflow-batch verifieras likadant.
- Provenance-predikatet pinnat på källa: repo `AI-Riksarkivet/htrflow`,
  workflow-fil, release-tagg; publicerat till registret bredvid imagen.
- Konsumentsidan: htrflow-batch-wrapperns bygge verifierar basimagens
  provenance före `FROM` (`gh attestation verify`/`slsa-verifier` i
  dagger-bygget), och ai-dev:s Kyverno `verifyImages` för
  `docker.io/riksarkivet/htrflow*` kräver attestationen med rätt utgivare
  och subjekt.
- Dokumenterat i htrflows docs: hur man verifierar en image för hand,
  med det exakta kommandot och förväntat utfall.

## Klart när

- [ ] `gh attestation verify oci://docker.io/riksarkivet/htrflow:<tagg>
      --owner AI-Riksarkivet` rapporterar SLSA Build Level 3.
- [ ] En image byggd från en fork eller ett annat workflow avvisas av
      Kyverno i dev-klustret; den riktiga släpps igenom.
- [ ] htrflow-batch-wrapperns publish-bygge misslyckas om basimagens
      provenance saknas eller pekar på fel repo.
