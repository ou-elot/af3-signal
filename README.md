# AF3 structural-signal pilot

A small, reproducible Python pipeline comparing **tau first, then amyloid-beta** AF3 monomer predictions with patient-derived fibril chains, using exact-composition sequence shuffles as an operational sequence-order null. This is a descriptive research pilot, not a disease predictor. A null result is a successful experiment.

## Current implemented status

Real reference files have been downloaded, cached unchanged, hashed, parsed, and sequence-verified. The full pilot has **36 constructs / 108 construct-seed jobs** across two proteins; the tau-only pilot has **18 / 54**. There are **zero imported AF3 outputs**. Prediction-dependent results are explicitly **BLOCKED/PENDING**, without simulated findings.

Start with the [verified status snapshot](reports/pilot_status.md). After running the pipeline, the complete generated report is at `runs/pilot/reports/research_report.md`. Generated artifacts are available locally and ignored by git; the commands below regenerate them.

| Protein | Reference | Derived representative coordinate coverage | Both-angle positions |
|---|---|---|---|
| Tau | 5O3L, primary PHF | canonical 306–378, 73/73 | 71 |
| Tau | 5O3T, sensitivity SF | canonical 306–378, 73/73 | 71 |
| A-beta42 | 7Q4B, primary Type I | A-beta 9–42, 34/42 | 32 |
| A-beta42 | 7Q4M, sensitivity Type II | A-beta 12–42, 31/42 | 29 |

Label chain `A` is the deterministic representative in these four current files. Other chains remain in QC tables, never counted as biological replicates. A-beta coverage is measured from each file, not assumed in the implementation.

## Setup

Use Python **3.12** (verified on Windows). From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

If activation is unavailable, use `.\.venv\Scripts\python.exe` in place of `python`. On Linux/macOS activate with `source .venv/bin/activate`. Direct package, test, and build dependencies are pinned in `pyproject.toml`; `requirements-lock.txt` records the exact verified environment including transitive dependencies. To recreate it, install `python -m pip install -r requirements-lock.txt` before the editable project install.

Commands use paths relative to the project working directory. Keep that working directory consistent. Prediction file paths inside CSVs are relative to the configured **run directory**, unless explicitly absolute.

## Run the pilot

### Tau first

```powershell
python -m af3_signal fetch-references --config configs/tau.yaml
python -m af3_signal prepare-inputs --config configs/tau.yaml
python -m af3_signal validate-predictions --config configs/tau.yaml --manifest runs/tau/predictions/manifest.csv
python -m af3_signal analyze --config configs/tau.yaml
python -m af3_signal report --config configs/tau.yaml
```

### Tau and A-beta

```powershell
python -m af3_signal fetch-references --config configs/pilot.yaml
python -m af3_signal prepare-inputs --config configs/pilot.yaml
python -m af3_signal validate-predictions --config configs/pilot.yaml --manifest runs/pilot/predictions/manifest.csv
python -m af3_signal analyze --config configs/pilot.yaml
python -m af3_signal report --config configs/pilot.yaml
```

An empty prediction manifest is valid: analysis produces the reference/mask tables, explicit pending comparisons, and a status report. This is the expected current path. Both reference-only and imported-output analysis run **without network access**. `fetch-references --offline` rechecks the raw cache. Online fetching resumes `.part` files when the server honors HTTP Range; committed raw files are never silently replaced. Hash/source mismatches are errors. Retrieval timestamps use the machine's UTC clock, not a hardcoded date.

### Small smoke run

Use the same five commands with `--config configs/smoke.yaml`. This prepares tau + one shuffle, three contexts, one seed: **6 jobs**. Run `fetch-references` once for that run directory. No GPU or inference is needed to smoke-test the pending path. Actual prediction import can then test the same path on real outputs.

## Frozen experiment design

`configs/pilot.yaml` defines the complete sequence and reference design, two 20-residue candidate-disordered flank pairs, the exact shuffle RNG seed, five unique permutations per protein, three prediction seeds, confidence/coverage thresholds, beta region, and contact definition. Flank disorder is an **unchecked assumption**. The same shuffled segment is reused across all three contexts, including fair unflanked shuffle controls.

`configs/tau.yaml` and `configs/smoke.yaml` inherit from `pilot.yaml`. A stable protein-specific RNG stream means enabling A-beta does not alter tau shuffles. Duplicated or native-equal permutations are rejected. Model inputs have neutral hashed IDs, not protein names.

Each run freezes a resolved `config.lock.json`; changing the design or thresholds requires a **new `run_dir`**. Set model version/checkpoint before inference if known; otherwise the current explicit `unknown_pending_inference` values are resolved from actual import metadata. Actual known MSA/template settings must agree with the frozen plan. Separate model versions/checkpoints/settings require separate runs, not pooled results.

The scheduling estimate is five diffusion samples per seed (540 possible samples in the full pilot), but **actual imported samples are counted separately**. The smoke estimate is one; actual installations may produce a different count. A job becomes `IMPORTED` when at least one sample exists. This status does not certify that every diffusion sample has been imported: check `actual_samples` against the actual inference output, not just the scheduling estimate. Import every available sample, not only the top-ranked one.

## AF3 execution and manual import

### Available inputs

Within a run, `inputs/constructs.fasta` and one FASTA per construct provide sequences. `inputs/af3/*.json` provides DeepMind local-AF3 **dialect `alphafold3`, version 1** inputs with explicit model seeds, chain `A`, `unpairedMsa: ""`, `pairedMsa: ""`, and `templates: []`. This is a deliberately **MSA-free, template-free** pilot; these settings affect interpretation.

No inference engine, credentials, website scraper, or assumed API is included. AF3 inference requires your own installation/authorized service and its actual model access. A browser ChatGPT Plus login is not AF3 inference access.

For a compatible local AF3 installation on a supported GPU host, use its documented environment, for example from its checkout/container:

```bash
python run_alphafold.py \
  --input_dir=/absolute/project/runs/pilot/inputs/af3 \
  --output_dir=/absolute/project/runs/pilot/predictions/local_af3 \
  --model_dir=/path/to/actual/model_parameters \
  --run_data_pipeline=false \
  --run_inference=true
```

Check that installation's `python run_alphafold.py --help` and input/output documentation; save the command, git revision, checkpoint identifier/hash, actual settings, date, and output terms/provenance. Version-1 input is the initial documented local format. Server JSON has a different dialect; when using a server manually, supply the FASTA sequences and preserve its actual seeds/settings/metadata. If a service cannot provide the prescribed seeds or settings, preregister an appropriate separate run rather than relabeling them. Do not silently use AF2, ESMFold, or Boltz.

### Fill the prediction manifest

Preparation creates `<run_dir>/predictions/manifest.csv` with headers. Copy the row layout from [`predictions/manifest.example.csv`](predictions/manifest.example.csv) and replace all placeholders using **real output files**. One row is one actual seed/sample, not one job or one residue. Do not import the top-ranked root-level file again when it duplicates a seed/sample file.

| Field | Meaning |
|---|---|
| `sample_id` | Unique metadata label for one actual output |
| `construct_id` | Exact ID from `inputs/constructs.csv` |
| `seed`, `sample_index` | Actual integer seed and diffusion sample index; seed must be configured |
| `cif_path` | Actual AF3 mmCIF path, relative to `run_dir` (or absolute) |
| `confidence_json` | Optional **full atom** confidence JSON, not summary-only JSON; leave empty if absent |
| `chain_id` | **Label** chain ID from mmCIF; emitted inputs request `A` |
| `label_offset` | `construct_position = label_seq_id - label_offset`; normally `0`, never an author-number offset |
| `model_name` | `alphafold3`; this adapter rejects other families |
| `model_version`, `checkpoint` | Actual installation/revision and checkpoint identity; explicitly `unknown` if unavailable |
| `input_sha256` | Exact generated input JSON hash from `inputs/constructs.csv`; preserve the original input as the design artifact |
| `msa_settings`, `template_settings` | Actual values, normally `none` and `disabled` for this design; explicitly `unknown` when unavailable |
| `inference_date`, `inference_source` | Actual date and installation/service/command provenance; explicit `unknown` allowed |
| `confidence_source` | `af3_bfactor` when documented as AF3 pLDDT, `af3_json` for full JSON alone, or `missing` |
| `confidence_evidence` | Documentation/output-metadata evidence for interpreting atom B values as AF3 pLDDT; required for explicit `af3_bfactor` |

For a manual service run, the generated JSON hash identifies the frozen sequence/seed design, not proof that the server executed that JSON. Retain the actual submitted artifact and settings in `inference_source`, and record differences/unknowns honestly.

The adapter validates every modeled residue against its expected **construct** sequence, including flanks and shuffled identities. Missing residues remain explicit missing rows: residue numbers are never compressed. Native canonical mapping verifies native identity; shuffled-to-reference comparison is expressly position-mapped, not residue-identity-matched. The current adapter deliberately handles a **single-chain protein monomer per output** and rejects ambiguous insertion/duplicate labels or multi-model/multichain predictions with actionable errors.

### Confidence conventions

- Full local-AF3 JSON uses `atom_plddts` and `atom_chain_ids` in mmCIF atom order. Counts and chain order must match; values must be 0–100 or missing.
- If mmCIF pLDDT is documented, JSON is cross-checked against its B values (0.11 tolerance for serialization rounding). A mismatched sample/order fails validation. JSON alone lacks per-atom names; count/chain checks cannot prove same-chain atom order. Preserve the original paired files.
- B values are accepted only with AF3+pLDDT metadata or an explicit documented AF3-output attestation. Arbitrary experimental/PDB B-factors are never guessed to be confidence.
- Residue pLDDT is the mean of **N, CA, C**, only when all three atoms have confidence. Missing confidence remains missing and is flagged in the normalized manifest. Primary geometry can still be analyzed; missing confidence makes the relevant secondary masks unavailable. Summary JSON does not provide residue confidence and is rejected if supplied as the full JSON.

After filling the CSV, rerun validation, analysis, and reporting with the same `--config`. Validation failures return nonzero with the sample ID and reason. Actual source hashes, confidence conventions, unknown provenance, both chain-numbering schemes, and all construct/canonical positions are exported.

## Methods and endpoints

### Reference ingestion and numbering

Original RCSB mmCIF bytes and entry metadata are cached under `data/raw/`, with URL, retrieval UTC, SHA-256, and response headers in sidecars. Primary paper DOIs are checked against RCSB metadata. AF3 citation metadata and its official input/output documentation are also cached with hashes.

Tau 306–378 is checked against **UniProt `P10636-8` Tau-F, 441 residues**. UniProt's default `P10636` sequence is longer, so using its raw positions would be incorrect. A-beta1–42 is verified against APP `P05067`, precursor positions 672–713; both numbering systems are exported. Unique exact sequence anchoring to mmCIF entity-polymer sequences establishes label-to-canonical mapping, then modeled amino-acid identities are verified. Unsupported/ambiguous anchoring fails instead of guessing author numbering.

Representative selection: highest both-angle coordinate coverage, then lexicographic label chain. Altloc policy: highest summed N/CA/C occupancy for a single residue conformer, lexicographic tie-break, plus shared blank-altloc atoms. Conflicting identities/insertion ambiguities are angle-ineligible.

### Primary endpoint

Compute phi/psi from N/CA/C and neighboring residues. Do not bridge missing residues/atoms, nonconsecutive label or canonical positions, insertion ambiguities, or C–N peptide distances > 1.8 Å. Terminals and degenerate angles remain missing with reasons.

```text
delta(a,b) = abs(((a-b+180) % 360)-180)
error_i = (delta(phi_pred,phi_ref) + delta(psi_pred,psi_ref))/2
```

**Primary fixed mask:** reference positions with valid phi **and** psi, intersected with internal segment positions before scoring any prediction. This gives identical eligibility across flank contexts; target terminals cannot benefit merely from added flanks. Every predetermined position is required for a primary score. An incomplete output gets a missing primary score, not an easier available-residue mean; coverage below **90%** is additionally flagged. Minimum eligible mask size is **10**.

**Common-mask sensitivity:** intersect valid positions across all imported native/control samples in each protein/reference/context, then score on this shared mask if at least 10 remain. The mask depends on the imported set and is explicitly secondary; import all outputs before interpreting it.

### Secondary endpoints and QC

- **Low confidence:** pLDDT **< 50**, defined before outcomes. Export known-confidence counts, missing counts, fraction low among known target residues, and variable-mask scores. No confidence values are forced low.
- **Paired-position sensitivity:** apply each native sample's low-confidence eligible positions to controls at the same seed/sample index. Require at least 10 and complete geometry on that mask; absent native matches are flagged. Native and shuffle residue identities need not match.
- **Beta occupancy:** inclusive phi `[-180,-90]`, psi `[90,180]` degrees, descriptive only. Denominator is available both-angle primary-eligible residues; consult coverage before comparing.
- **Longer range:** proper-rotation Kabsch C-alpha RMSD and intrachain contact recall/Jaccard, using the entire predetermined reference target C-alpha set. Contact: distance **< 8 Å**, sequence separation **> 3**. Missing predicted C-alpha positions withhold these scores. Empty reference contacts yield missing overlap with a reason. No interchain contacts are included.
- Flanks never enter target geometry scores. Their mean confidence and available count are separate QC fields.
- All representative and repeated-chain reference angles retain missingness reasons; repeated chains do not increase n. DSSP is not required or included in the core implementation.

### Aggregation and interpretation

Average samples within each seed, then average seeds equally within each construct. Any missing imported sample score invalidates that construct/metric aggregate. Export planned versus imported seed coverage. Compare the native value to the distribution over **independent shuffled constructs**, separately by protein, reference, and flank. Report every individual construct and native-minus-shuffle effect (negative favors native), not a selected best reference/seed/threshold. Partial imports are explicitly labeled; there are no formal p-values.

**Important limits:** low pLDDT measures structural uncertainty, not aggregation propensity. AF3 may hallucinate order. Seeds are computational replicates, not conformational populations. Fibrils are assembled structures, so monomer mismatch does not disprove aggregation propensity. Flanks do not hide identity or remove training exposure. Shuffles change residue preferences by position and form a conditional sequence-order null, not a disease-specific null. Fibrils are not proof of a causal disease structure; Type II A-beta also occurs outside Alzheimer's. Residues, repeated chains, seeds, flanks, and related polymorphs are not independent biological samples. Generic sequence/structure preferences, training exposure, and artifacts can explain similarity.

### Exposure audit

The AF3 paper's Data availability section reports a PDB snapshot downloaded **2023-01-12**, with templates downloaded **2022-09-28**. The four references were released in 2017 or November 2021, before this snapshot. That establishes possible exposure, **not verified inclusion or a cutoff for your actual checkpoint**. The audit retains unknown training-cutoff/inclusion fields and actual imported model/template provenance. Disabling inference templates does not remove training exposure. This project supports no unseen-structure generalization, disease prediction, or hidden-pathological-signal claim.

## Artifact map

Everything generated is under the selected `runs/<name>/`:

```text
config.lock.json                       frozen resolved experiment
data/raw/                              unchanged mmCIF, RCSB metadata, FASTA, source docs + sidecars
data/processed/reference_mapping.csv   all chains: author/label/insertion/AA/canonical/APP mappings
data/processed/reference_angles.csv    all chains: phi/psi and missingness reasons
data/processed/reference_coverage.csv  representative flag and exact resolved positions
data/processed/reference_provenance.csv
data/processed/source_provenance.csv
data/processed/exposure_audit.csv
inputs/constructs.csv                  sequences, flanks, RNG seeds, offsets, input hashes
inputs/construct_reference_mapping.csv native and shuffled operational mappings per reference
inputs/jobs.csv                        original planned construct-seed jobs
inputs/design.json                     design hash and predefined null/mask/aggregation
inputs/constructs.fasta                identity-neutral headers
inputs/af3/*.json                      optional supported local inference inputs
predictions/manifest.csv               researcher-supplied real output registration
data/processed/predictions_normalized.csv
data/processed/prediction_mapping.csv  including explicit missing residues and flank positions
data/processed/prediction_angles.csv   all construct residues, confidence and missingness
results/predetermined_masks.csv
results/reference_summary.csv
results/per_residue.csv                prediction/reference angles, confidence, errors, masks
results/sample_scores.csv              angular/low-confidence/contact/RMSD/QC by sample/reference
results/construct_scores.csv           equal-seed-weighted native and independent shuffle values
results/comparisons.csv                all configured angular effects and status
results/pending_jobs.csv               actual output counts per job
figures/*.{png,svg}                    equal-axis Ramachandran, confidence, errors, construct scores
reports/research_report.md             actual results or explicit preparation/status report
reports/artifact_hashes.json           hashes of generated deliverables
results/analysis_state.json            input/artifact hashes checked before reporting
```

With no imported predictions, prediction/sample tables have headers and zero data rows; comparisons have explicit pending rows. Ramachandran figures contain only real reference points. Confidence/error/score panels say PENDING; no synthetic AF3 outputs enter the research run. Figures distinguish reference crosses, native blue points/lines, and shuffled grey points/lines (orange construct scores). Overlaid local angles do not establish identical folds.

Reporting verifies the analysis receipt, manifest, source files, and analyzed artifact hashes. If an input or analyzed table changes, rerun `analyze` before `report`; a stale result cannot silently become a current report.

## Verification

```powershell
python -m pytest -q
python -m af3_signal fetch-references --config configs/pilot.yaml --offline
python -m af3_signal analyze --config configs/pilot.yaml
python -m af3_signal report --config configs/pilot.yaml
```

Tests are network-independent and use tiny, transparent **SYNTHETIC SOFTWARE TEST ONLY** coordinate curves generated in pytest temporary directories. They test wraparound, true missingness, known torsion orientation, identical-structure errors, offsets and missing residues, chain breaks, altloc/insertion handling, sequence mismatches, exact-composition shuffle reproducibility, context matching, config freezing, confidence absence/alignment errors, source/input provenance errors, empty contacts/masks, proper-rotation alignment, equal seed weighting, partial/common masks, complete import analysis, and the pending report path. Fixtures are not experimental results or purported AF3 inference.

## Sources

- Fitzpatrick et al. (2017), *Cryo-EM structures of tau filaments from Alzheimer's disease*. [10.1038/nature23002](https://doi.org/10.1038/nature23002). RCSB [5O3L](https://www.rcsb.org/structure/5O3L), [5O3T](https://www.rcsb.org/structure/5O3T).
- Yang et al. (2022), *Cryo-EM structures of amyloid-beta 42 filaments from human brains*. [10.1126/science.abm7285](https://doi.org/10.1126/science.abm7285). RCSB [7Q4B](https://www.rcsb.org/structure/7Q4B), [7Q4M](https://www.rcsb.org/structure/7Q4M).
- Abramson et al. (2024), *Accurate structure prediction of biomolecular interactions with AlphaFold 3*. [10.1038/s41586-024-07487-w](https://doi.org/10.1038/s41586-024-07487-w). Official [input](https://github.com/google-deepmind/alphafold3/blob/main/docs/input.md) and [output](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md) documentation.
