# Verified pilot status

This is a small, tracked status snapshot. Regenerate the complete report with the README commands; it is written to `runs/pilot/reports/research_report.md`.

## Available evidence

The four original RCSB mmCIF files were retrieved and sequence-verified. Tau numbering uses the 441-residue UniProt `P10636-8` isoform; A-beta1–42 is checked against APP `P05067` residues 672–713.

| Reference | Protein | Representative label chain | Resolved canonical positions | Both-angle residues |
|---|---|---|---|---:|
| 5O3L | Tau | A | 306–378 | 71 |
| 5O3T | Tau | A | 306–378 | 71 |
| 7Q4B | A-beta | A | 9–42 | 32 |
| 7Q4M | A-beta | A | 12–42 | 29 |

Repeated chains remain coordinate QC, not independent biological samples. Reference coverage was derived from the files rather than hardcoded.

## Prepared experiment

- Full pilot: 36 constructs, 108 construct-seed jobs.
- Per protein: native plus five composition-preserving unique shuffles, three fixed flank contexts, three prediction seeds.
- Tau smoke pilot: six constructs and six construct-seed jobs.
- Actual imported AF3 output samples: **0**.
- Prediction-dependent comparisons: **BLOCKED/PENDING**.
- Verification: **17 tests passed**; real-reference tau, full-pilot, and smoke preparation/status paths completed. Cached-reference analysis works without network access.

The official AF3 source has also been retrieved locally at commit `03f4f48867748c164ae6d5144ae0529477df02e3`; it is not bundled with this repository. Model parameters and a working inference runtime are still required.

## Interpretation

No native-versus-shuffle result, positive or null, has been established. Low pLDDT measures structural uncertainty, not aggregation propensity or disease probability. A monomer prediction need not reproduce an assembled fibril. Position-mapped shuffles are a sequence-order null, not a disease-specific null. Seeds, residues, repeated chains, contexts, and related references are not independent biological samples.

All four references predate the AF3 paper's reported 2023-01-12 PDB snapshot. Actual checkpoint training inclusion/cutoff remains unknown; disabling templates does not remove training exposure. This pilot supports no unseen-structure generalization or disease-specific causal claim. Type II A-beta also occurs outside Alzheimer's disease.

Complete the experiment by running real AF3 inference under the frozen design, registering all available samples with their actual metadata, and rerunning validation, analysis, and reporting. A null result is a successful experiment.
