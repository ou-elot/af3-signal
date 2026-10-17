"""Actual-data figures and a status-aware Markdown research report."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import digest, initialize, sha256, table, utcnow, write_json
from .predictions import resolve_source


LIMITATIONS = """- Low pLDDT is structural uncertainty, not aggregation propensity or disease probability.
- AF3 can hallucinate order in disordered regions. Beta-like geometry alone is not evidence of biological signal.
- Seeds and output samples are computational replicates, not physical ensemble populations. Residues, repeated fibril chains, seeds, flanks, and related references are not independent biological samples.
- Fibrils are assembled states. Monomer mismatch does not establish absence of aggregation propensity; intrachain contacts exclude assembly-dependent interchain geometry.
- Flanks change sequence context; their disorder is assumed, not verified. They do not obscure sequence identity or remove training exposure.
- Experimental references do not establish a disease-causing structure. Type II A-beta also occurs outside Alzheimer's and is not disease-specific.
- Position-mapped composition-preserving shuffles are an operational sequence-order null. Residue preferences differ at each position; this is not a disease-specific null.
- Generic sequence/structure preferences, training exposure, and artifacts can produce reference similarity. Disabling templates does not remove training exposure.
- Five shuffles provide poorly resolved inference. This pilot reports descriptive effects without formal significance or an unseen-structure generalization claim.
- All configured references, contexts, seeds, and thresholds must be reported; no best-match selection. A null result is a successful experiment.
"""


def markdown_table(frame):
    if frame.empty:
        return '_No rows available._\n'
    copy = frame.copy().fillna('—')
    columns = list(copy.columns)
    def cell(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    return ('| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n' +
            '\n'.join('| ' + ' | '.join(cell(v) for v in row) + ' |' for row in copy.itertuples(index=False, name=None)) + '\n')


def save(fig, folder, name, test_only=False):
    if test_only:
        fig.suptitle('SYNTHETIC SOFTWARE TEST ONLY — NOT RESEARCH DATA', fontsize=8, color='red')
    fig.tight_layout()
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'{name}.{suffix}', dpi=160)
    plt.close(fig)


def pending(ax, title):
    ax.text(.5, .5, 'PENDING: no imported AF3 predictions', ha='center', va='center', transform=ax.transAxes)
    ax.set_title(title)


def figures(config):
    root = initialize(config)
    refs = pd.read_csv(root / 'data/processed/reference_angles.csv')
    pred = pd.read_csv(root / 'data/processed/prediction_angles.csv')
    agg = pd.read_csv(root / 'results/construct_scores.csv')
    errors = pd.read_csv(root / 'results/per_residue.csv')
    constructs = pd.read_csv(root / 'inputs/constructs.csv').set_index('construct_id')
    folder = root / 'figures'
    for protein in config['enabled_proteins']:
        for ref in config['proteins'][protein]['references']:
            rid = ref['id']
            rr = refs[(refs.reference == rid) & refs.representative]
            for context in config['flanks']:
                fid = context['id']
                ids = set(constructs[(constructs.protein == protein) & (constructs.flank_id == fid)].index)
                pp = pred[pred.construct_id.isin(ids) & pred.is_target]
                tag = f'{protein}_{rid}_{fid}'
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(rr.phi, rr.psi, marker='x', color='black', s=24, label=f'Experimental {rid}, representative chain')
                for variant, color, marker in [('shuffle', '0.65', '.'), ('native', '#0072B2', 'o')]:
                    cc = set(constructs[constructs.variant == variant].index)
                    ss = pp[pp.construct_id.isin(cc)]
                    if len(ss):
                        ax.scatter(ss.phi, ss.psi, s=12, alpha=.5, marker=marker, color=color, label=f'AF3 {variant}, all imported samples')
                ax.set(xlim=(-180, 180), ylim=(-180, 180), xlabel='phi (degrees)', ylabel='psi (degrees)',
                       title=f'{protein} / {rid} / {fid}\nLocal angles only; overlap does not imply identical folds')
                ax.set_aspect('equal', adjustable='box')
                ax.legend(fontsize=7)
                save(fig, folder, tag + '_ramachandran', config.get('software_test_only', False))
                fig, ax = plt.subplots(figsize=(8, 4))
                if pp.empty:
                    pending(ax, f'{tag}: target confidence')
                else:
                    for sid, group in pp.groupby('sample_id'):
                        variant = constructs.loc[group.construct_id.iloc[0], 'variant']
                        ax.plot(group.canonical_position, group.plddt, alpha=.5, color='#0072B2' if variant == 'native' else '0.65', lw=1)
                    ax.axhline(config['analysis']['low_confidence_threshold'], color='black', ls='--', lw=1)
                    ax.set_title(f'{tag}: AF3 pLDDT (native blue; shuffles grey)')
                    if not pp.plddt.notna().any():
                        ax.text(.5, .5, 'Per-residue confidence MISSING', transform=ax.transAxes, ha='center')
                ax.set(xlabel='Canonical target residue', ylabel='Mean N/CA/C pLDDT', ylim=(0, 100))
                save(fig, folder, tag + '_confidence', config.get('software_test_only', False))
                fig, ax = plt.subplots(figsize=(8, 4))
                ee = errors[(errors.reference == rid) & errors.construct_id.isin(ids)]
                if ee.empty:
                    pending(ax, f'{tag}: residue-specific angular errors')
                else:
                    for sid, group in ee.groupby('sample_id'):
                        variant = constructs.loc[group.construct_id.iloc[0], 'variant']
                        ax.plot(group.canonical_position, group.error, alpha=.5, color='#0072B2' if variant == 'native' else '0.65')
                ax.set(xlabel='Canonical target residue', ylabel='Mean circular phi/psi error (degrees)', ylim=(0, 180))
                save(fig, folder, tag + '_errors', config.get('software_test_only', False))
                fig, ax = plt.subplots(figsize=(7, 4))
                ss = agg[(agg.reference == rid) & (agg.flank_id == fid) & (agg.protein == protein) & (agg.metric == 'primary_error')]
                if ss.empty:
                    pending(ax, f'{tag}: primary construct scores')
                else:
                    for x, variant, color in [(0, 'native', '#0072B2'), (1, 'shuffle', '#D55E00')]:
                        values = ss[ss.variant == variant].mean_error.dropna()
                        offsets = np.linspace(-.1, .1, len(values)) if len(values) > 1 else np.zeros(len(values))
                        ax.scatter(x + offsets, values, color=color, s=40)
                    ax.set_title(f'{tag}: one point per construct\nSamples averaged within seed, then seeds within construct')
                    if not ss.mean_error.notna().any():
                        ax.text(.5, .5, 'INSUFFICIENT fixed-mask coverage', ha='center', transform=ax.transAxes)
                ax.set(xticks=[0, 1], xticklabels=['Native', 'Independent shuffles'], xlim=(-.5, 1.5),
                       ylabel='Mean angular error (degrees; lower is better)', ylim=(0, 180))
                save(fig, folder, tag + '_scores', config.get('software_test_only', False))


def report(config):
    root = initialize(config)
    needed = root / 'results/sample_scores.csv'
    if not needed.exists():
        raise ValueError('Run analyze before report')
    receipt_path = root / 'results/analysis_state.json'
    if not receipt_path.exists():
        raise ValueError('Missing analysis receipt; rerun analyze before report')
    receipt = json.loads(receipt_path.read_text())
    if receipt['config_sha256'] != digest(config) or sha256(root / receipt['manifest_path_relative_to_run']) != receipt['manifest_sha256']:
        raise ValueError('Prediction manifest/config changed since analysis; rerun analyze before report')
    for relative, expected in receipt['analysis_artifact_sha256'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'Analysis artifact changed: {relative}; rerun analyze before report')
    sources = pd.read_csv(root / 'data/processed/predictions_normalized.csv', keep_default_na=False)
    for _, sample in sources.iterrows():
        for path_column, hash_column in [('cif_path', 'coordinate_sha256'), ('confidence_json', 'confidence_sha256')]:
            if sample[path_column] and sha256(resolve_source(root, sample[path_column])) != sample[hash_column]:
                raise ValueError(f'Prediction source changed for {sample.sample_id}; rerun analyze before report')
    figures(config)
    coverage = pd.read_csv(root / 'data/processed/reference_coverage.csv')
    provenance = pd.read_csv(root / 'data/processed/reference_provenance.csv')
    audit = pd.read_csv(root / 'data/processed/exposure_audit.csv')
    normalized = pd.read_csv(root / 'data/processed/predictions_normalized.csv')
    jobs = pd.read_csv(root / 'results/pending_jobs.csv')
    scores = pd.read_csv(root / 'results/sample_scores.csv')
    comparisons = pd.read_csv(root / 'results/comparisons.csv')
    angles = pd.read_csv(root / 'data/processed/reference_angles.csv')
    representatives = coverage[coverage.representative]
    reference_summary = []
    for rid, group in angles[angles.representative].groupby('reference'):
        from .metrics import beta_fraction
        a = config['analysis']
        reference_summary.append({'reference': rid, 'both_angle_residues': int((group.phi.notna() & group.psi.notna()).sum()),
                                  'phi_missing': int(group.phi.isna().sum()), 'psi_missing': int(group.psi.isna().sum()),
                                  'descriptive_beta_fraction': beta_fraction(group.phi, group.psi, a['beta_phi'], a['beta_psi'])})
    table(root / 'results/reference_summary.csv', reference_summary)
    actual = len(normalized)
    pending_count = int((jobs.status == 'PENDING').sum())
    status = 'BLOCKED / PENDING AF3 INFERENCE' if not actual else ('PARTIAL IMPORT' if pending_count else 'IMPORTED; inspect coverage and comparison status')
    text = f"""# AF3/fibril structural-signal pilot — preparation and research status

{'**SYNTHETIC SOFTWARE TEST ONLY — NOT RESEARCH DATA**' if config.get('software_test_only') else ''}

Generated UTC: {utcnow()}<br>
Configuration SHA-256: `{digest(config)}`<br>
Status: **{status}**

## Question and available evidence

Do original-sequence AF3 predictions have lower residue-specific backbone angular error to experimental fibril chains than exact-composition shuffled controls? Tau is prepared first; A-beta is configured separately. No model is trained.

There are **{len(jobs)} planned construct-seed jobs**, **{actual} actual imported output samples**, and **{pending_count} jobs with no imported samples**. Expected diffusion samples in scheduling manifests are estimates, not observations. Individual prediction samples and seeds never increase biological n.

{'No AF3 predictions have been supplied. Native-versus-control, low-confidence, and longer-range prediction results are BLOCKED/PENDING. No positive or null biological finding can yet be inferred.' if not actual else 'Results below are descriptive. PARTIAL_OR_INSUFFICIENT comparisons must not be interpreted as complete pilot outcomes. Inspect every configured comparison and the per-sample QC tables.'}

## Real reference coverage

One representative chain per reference is chosen by largest valid phi/psi coverage, then lexicographic label chain ID. Other chains remain in mapping/angle tables for coordinate QC, not sample-size inflation. Both author and label identifiers are retained; canonical sequence anchoring verifies identities rather than guessing chain IDs or author numbering. A-beta coverage is derived from coordinates.

{markdown_table(representatives[['protein', 'reference', 'role', 'label_chain', 'resolved_residues', 'target_length', 'eligible_both_angles']])}

{markdown_table(pd.DataFrame(reference_summary))}

Coverage and angle missingness are in `data/processed/reference_coverage.csv`, `reference_angles.csv`, and `results/predetermined_masks.csv`. Reference beta occupancy is descriptive, not a prediction result.

## Provenance and exposure

{markdown_table(provenance[['reference', 'release_date', 'doi', 'retrieved_utc', 'coordinate_sha256']])}

The AF3 paper reports a PDB snapshot downloaded 12 January 2023 and a template snapshot downloaded 28 September 2022. A snapshot is not a verified cutoff or proof that a particular entry was included. Actual inference version/checkpoint must be supplied in the import manifest; its exact training cutoff may remain unknown. Compare each release date in the audit rather than assuming a novel reference. This is potential reference exposure; an unseen-structure generalization claim is unsupported. Source documents and retrieval metadata are cached under `data/raw/sources/`.

{markdown_table(audit[['reference', 'predates_paper_snapshot', 'actual_checkpoint_training_cutoff', 'training_inclusion', 'actual_template_exposure']])}

Actual inference provenance and its unknown fields are retained in `data/processed/predictions_normalized.csv`.

## Preregistered methods

- Exact compositions are permuted without duplicate/native controls, using recorded independent per-protein RNG streams. The same shuffled segments are reused across fixed contexts. Flank sequences, offsets, seeds, and hashes are in `inputs/constructs.csv` and the frozen config.
- Alternate conformers: select one residue-level altloc by highest summed backbone occupancy, lexicographic tie-break, retaining shared blank-altloc atoms. Insertions or conflicting identities are angle-ineligible. Missing atoms, nonconsecutive labels/canonical positions, peptide C–N distances above {config['analysis']['peptide_bond_max_angstrom']} Å, degenerate geometry, and terminals yield missing angles with reasons; zero is never substituted.
- Circular difference: `abs(((a-b+180) % 360)-180)`; residue error is mean phi/psi difference.
- Primary fixed mask: reference positions with both angles valid, restricted to internal segment positions, predetermined before predictions. Every sample must provide all positions for a primary score; otherwise it is withheld, even if above the {config['analysis']['min_coverage']:.0%} QC threshold. All-sample intersection-mask sensitivity is exported separately by protein/reference/context. Minimum score mask size: {config['analysis']['min_residues']}.
- Confidence: full AF3 atom JSON, or explicitly documented AF3 pLDDT B-factor metadata/attestation. Per residue, mean N/CA/C only when all three confidence values exist. Missing confidence remains missing. Confidence JSON is checked for atom counts/chain order and cross-checked against documented B-factor pLDDT where available; JSON alone cannot prove same-chain atom identity/order.
- Secondary low confidence: pLDDT < {config['analysis']['low_confidence_threshold']}, with mask sizes, known-confidence fraction, and missing counts. Native low-confidence positions are applied to matched controls at the same seed/sample index; missing native samples are flagged. Flanks are excluded from target scores; their confidence summaries are QC only.
- Descriptive beta region: phi in {config['analysis']['beta_phi']}, psi in {config['analysis']['beta_psi']} degrees (inclusive). Intrachain C-alpha contacts use distance < {config['analysis']['contact_distance_angstrom']} Å and sequence separation > {config['analysis']['contact_min_sequence_separation']}; no reference contacts gives missing overlap, not zero. RMSD uses proper-rotation Kabsch alignment. Longer-range metrics require the complete fixed reference C-alpha mask.
- Average samples within each seed, then give seeds equal weight within each construct. A missing score invalidates that construct/metric aggregate; pending jobs and partial seed coverage are explicit. Compare native to independent shuffled constructs separately by protein/reference/context; negative native-minus-shuffle favors native. No formal p-values or population-level inference.

## All configured comparisons

{markdown_table(comparisons[['protein', 'reference', 'flank_id', 'metric', 'native_error', 'shuffle_mean', 'native_minus_shuffle', 'independent_shuffle_constructs', 'status']])}

## Coverage and robustness

{markdown_table(scores[['sample_id', 'reference', 'coverage', 'primary_status', 'common_mask_count', 'low_variable_count', 'native_low_paired_count', 'target_low_fraction_known', 'target_confidence_missing']]) if len(scores) else '_Prediction coverage and robustness analyses are pending._'}

Every imported value is exported in `results/per_residue.csv`, `sample_scores.csv`, and `construct_scores.csv`; job/sample completeness is in `pending_jobs.csv`. Per-context/reference PNG and SVG figures show reference angles now, and confidence/errors/construct distributions when supplied. Empty panels are explicitly pending, never simulated. Flank and seed robustness can only be assessed after these real outputs arrive.

## Confounds and interpretation limits

{LIMITATIONS}

## Conclusion and next required action

{'Reference ingestion, sequence mapping, exact-composition control preparation, and the pending analysis path are complete. The scientific question remains unanswered until real AF3 inference is imported.' if not actual else 'Only the descriptive comparisons shown above are supported. Neither local similarity nor a native advantage would establish a disease-specific or causal conformation; a null comparison is also informative.'}

Run the exact generated inputs using a documented AF3 installation or submit the FASTA sequences manually. Import **all samples**, including actual seeds/sample indices, input hashes, model version/checkpoint, MSA/template settings, inference date and source. Unknown provenance must be explicitly `unknown`. Follow the repository README, run `validate-predictions`, then `analyze` and `report`. Pending results must not be replaced with synthetic research figures.
"""
    path = root / 'reports/research_report.md'
    path.write_text(text, encoding='utf-8')
    outputs = [p for folder in ['data/processed', 'inputs', 'results', 'figures', 'reports']
               for p in (root / folder).rglob('*') if p.is_file() and p.name != 'artifact_hashes.json']
    write_json(root / 'reports/artifact_hashes.json', {p.relative_to(root).as_posix(): sha256(p) for p in sorted(outputs)})
    return path
