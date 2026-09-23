# FullSongEval anonymous audit source

This archive reproduces the text-label rule audit and its downstream genre checks. It contains no audio waveforms, generated songs, listener data, credentials, machine paths, or author identity.

## Environment and tests

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
PYTHONPATH=src:scripts pytest -q tests/test_musiccaps_gate_calibration.py tests/test_mtg_jamendo_metadata_calibration.py tests/test_calibration_prediction_replay.py tests/test_target_masked_table_derivation.py tests/test_random_ranking_baseline.py tests/test_frozen_score_label_null.py tests/test_mard_prospective_transfer.py
```

## Verify the packaged reports

```sh
python -m full_song_eval.musiccaps_gate_calibration --validate-report reports/musiccaps_gate_calibration.json
python scripts/derive_target_masked_table.py \
  --musiccaps reports/musiccaps_gate_calibration.json \
  --mtg-jamendo reports/mtg_jamendo_metadata_calibration.json \
  --output reproduced/target_masked_manuscript_summary.json
cmp reports/target_masked_manuscript_summary.json reproduced/target_masked_manuscript_summary.json
```

Expected SHA-256 digests:

- `reports/musiccaps_gate_calibration.json`: `26216b93a455edc6d84fdcd177959b1c7ba48b1862e511a77a1bbce171436088`
- `reports/mtg_jamendo_metadata_calibration.json`: `a6b6386d94a65e7cd2ea099f4fe809f0ccc2f7227c1a94a3d46ff491e708374b`
- `reports/target_masked_manuscript_summary.json`: `dcd2004a447a14622ef9a3e0b7a968b321bf0691749a1dad7ff4afcc20947a7c`
- `reports/random_ranking_baseline_calibration.json`: `a724202e4bb8ce0cdc1a716a50915b73fde3d2ad333db6be50e2a3d634ce895e`
- `reports/frozen_score_independent_label_null.json`: `ff588e545af330cc213a90948da4f045492cbba4692f24c3d3433581798cf601`
- `reports/mard_prospective_transfer.json`: `bc684532c9cbb7a93d894426dad215d11d9f28934fbfb2a09a6c22d55000dae4`
- `reports/mard_artist_overlap_audit.json`: `2aaaecf26c594a410817a47a7a1fe00afb1af5f85e322945ecf94a9f9e5a3dab`
- `reports/mard_artist_disjoint_robustness.json`: `67e70571d90ee11169367a267750d832e84cf8ab8e157a22ec674e9e52cbc990`

The `reports/musiccaps_stage1_summary.json`, `reports/musiccaps_proxy_experiment.json`,
`reports/musiccaps_representation_stress_test.json`,
`reports/musiccaps_hf_top16_summary.json`,
`reports/musiccaps_full_panel_labelscale_analysis.json`, and
`reports/musiccaps_hf_top1296_min5_statistical_stability.json` files support the
appendix's scope comparisons. They use different label inventories or F1-based
classification protocols and are not inputs to the tested macro-AUPRC decision.
The four Stage-1 input reports are under `reports/stage1/`, with their hashes
recorded in `reports/musiccaps_stage1_summary.json`. Recompute that summary with:

```sh
python -m full_song_eval.musiccaps_stage1_summary \
  --reports reports/stage1/aspect_top16_author10.json \
            reports/stage1/aspect_top1296_author10.json \
            reports/stage1/audioset_top16_author10.json \
            reports/stage1/audioset_top211_author10.json \
  --output-json reproduced/musiccaps_stage1_summary.json \
  --output-md reproduced/musiccaps_stage1_summary.md \
  --output-tex reproduced/musiccaps_stage1_summary.tex
```
The summary code and public metadata manifest are included. Sanitized paths
and implementation digests can differ from the private run; compare numerical
design results and the four packaged input hashes.
`reports/ace_yue_automatic_feature_shift_diagnostic.json` supplies the
generator-family diagnostic cited only as a scope distinction.
`reports/full_song_eval_pilot_proxy_feature_comparison.csv` summarizes the
automatic pilot features, without listener or quality outcomes.

## Reproduce MusicCaps

The included public metadata manifest has 4,823 records and SHA-256 `dad59617a323a0aa56f09fb6c462a430df14c6184321b5851b93abcb14446a42`. No waveform is used.

```sh
python -m full_song_eval.musiccaps_gate_calibration \
  --manifest data/musiccaps_manifest.jsonl \
  --output-json reproduced/musiccaps_gate_calibration.json \
  --output-markdown reproduced/musiccaps_gate_calibration.md \
  --output-latex reproduced/musiccaps_gate_calibration_rows.tex \
  --top-label-count 16 --min-label-count 25 \
  --split-unit author_id --target-field aspect_list \
  --seed full-song-eval-gate-calibration-v1 \
  --code-commit 48f459cf6c663ebd1ccd67606a57c4108920286b
python -m full_song_eval.musiccaps_gate_calibration --validate-report reproduced/musiccaps_gate_calibration.json
```

## Reproduce MTG-Jamendo

Download the two official metadata files at revision `cafd8e20c265ed84f1e61f1c875327971f43a62f`. The metadata is licensed CC BY-NC-SA 4.0. No waveform is used.

```sh
mkdir -p data/mtg-jamendo reproduced
curl -L https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/cafd8e20c265ed84f1e61f1c875327971f43a62f/data/raw.meta.tsv -o data/mtg-jamendo/raw.meta.tsv
curl -L https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/cafd8e20c265ed84f1e61f1c875327971f43a62f/data/raw_30s_cleantags_50artists.tsv -o data/mtg-jamendo/raw_30s_cleantags_50artists.tsv
printf '%s  %s
' \
  bb1efa2876536cbe1a8b67db5dda15249ca144f0cff69e481b638803cbdc9aca data/mtg-jamendo/raw.meta.tsv \
  a30655f45f4a7b4f9dba2d6b51d78733be33687eae3e8b9795e2234d54ef4e56 data/mtg-jamendo/raw_30s_cleantags_50artists.tsv | shasum -a 256 -c -
python -m full_song_eval.mtg_jamendo_metadata_calibration \
  --metadata-tsv data/mtg-jamendo/raw.meta.tsv \
  --tags-tsv data/mtg-jamendo/raw_30s_cleantags_50artists.tsv \
  --manifest reproduced/mtg_jamendo_manifest.jsonl \
  --report reproduced/mtg_jamendo_metadata_calibration.json \
  --source-revision cafd8e20c265ed84f1e61f1c875327971f43a62f \
  --seed full-song-eval-mtg-jamendo-metadata-v1 \
  --top-label-count 16 --min-label-count 25
```

## Recompute prediction-level replay checks

The archive omits generated replay sidecars because they duplicate public metadata-derived scores. Recreate them locally and verify every fold, condition, support count, and target-masked macro AUPRC against the packaged reports:

```sh
python -m full_song_eval.calibration_prediction_replay \
  --musiccaps-manifest data/musiccaps_manifest.jsonl \
  --musiccaps-output reproduced/musiccaps_primary_prediction_replay.jsonl.gz \
  --musiccaps-report reports/musiccaps_gate_calibration.json \
  --mtg-manifest reproduced/mtg_jamendo_manifest.jsonl \
  --mtg-output reproduced/mtg_jamendo_primary_prediction_replay.jsonl.gz \
  --mtg-report reports/mtg_jamendo_metadata_calibration.json \
  --top-label-count 16 --min-label-count 25 \
  --verification-output reproduced/calibration_prediction_replay_verification.json
```

## Reproduce the finite-sample reference analysis

After generating the MTG-Jamendo manifest above, replay the frozen target-masked scorer and compute the exact strict-ranking and score-tie-aware random-label expectations:

```sh
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/build_random_ranking_baseline.py \
  --musiccaps-manifest data/musiccaps_manifest.jsonl \
  --mtg-manifest reproduced/mtg_jamendo_manifest.jsonl \
  --musiccaps-report reports/musiccaps_gate_calibration.json \
  --mtg-report reports/mtg_jamendo_metadata_calibration.json \
  --output reproduced/random_ranking_baseline_calibration.json
cmp reports/random_ranking_baseline_calibration.json reproduced/random_ranking_baseline_calibration.json
```

## Reproduce the score-frozen independent-label control

After generating and verifying both prediction replays above, run the fixed
200-draw control on the same held-out groups. It independently allocates each
label's observed positive count among records while retaining the original
scores. This is a conditional post-audit check, not an untouched-cohort test.

```sh
PYTHONPATH=src python scripts/run_frozen_score_label_null.py \
  --output reproduced/frozen_score_independent_label_null.json
```

Compare the accepted-draw counts and mean margins with
`reports/frozen_score_independent_label_null.json`. Packaged report hashes may
differ because the source paths in the archive have been anonymized.

## Reproduce the MARD transfer and artist-overlap checks

The authors' MARD genre-classification subset provides 1,300 album-review
records with genre labels under the project's MIT license. No raw review text
is in this archive. Download only the 28.4 MB source file at the pinned commit,
verify its digest, then run the recorded local protocol. The album folds are
disjoint by record, but a join to the creators' full metadata found 41 artist
IDs crossing folds. There is no independently sealed pre-run protocol record.

```sh
mkdir -p data/external/mard reproduced
curl -L https://raw.githubusercontent.com/sergiooramas/music-genre-classification/0cd900b02d90c021a9c5cacf2c5f3f6b35ef1e9c/dataset_classification.json -o data/external/mard/dataset_classification.json
printf '%s  %s\n' ea3c707f279281fb75f88549c738a437579246d2f8d8bd0e8e8a0e5cdccf2d6c data/external/mard/dataset_classification.json | shasum -a 256 -c -
PYTHONHASHSEED=0 PYTHONPATH=src python scripts/run_mard_prospective_transfer.py \
  --protocol configs/mard_transfer_protocol.json \
  --output reproduced/mard_prospective_transfer.json
cmp reports/mard_prospective_transfer.json reproduced/mard_prospective_transfer.json
```

The artist audit fetches only the 23.5 MB compressed metadata member of the
creators' MARD ZIP using HTTP byte ranges. It writes a lightweight artist-ID
mapping under ignored `data/`. The post-audit robustness run removes training
albums whose artist also appears in the held-out fold. It is a sensitivity
check, not an independent prospective study.

```sh
PYTHONPATH=src:scripts python scripts/audit_mard_artist_overlap.py
PYTHONPATH=src:scripts python scripts/run_mard_artist_disjoint_robustness.py
PYTHONPATH=src:scripts pytest -q tests/test_mard_artist_robustness.py
```

## Reproduce the changed-label and independent-genre diagnostics

The changed-label check is post-audit: it restricts each original control to held-out records whose selected-label vector changed. Run it after generating both verified prediction replays above:

```sh
PYTHONPATH=src python scripts/analyze_changed_label_controls.py --source MusicCaps \
  --manifest data/musiccaps_manifest.jsonl \
  --replay reproduced/musiccaps_primary_prediction_replay.jsonl.gz \
  --report reports/musiccaps_gate_calibration.json \
  --output reproduced/musiccaps_changed_label_controls.json
PYTHONPATH=src python scripts/analyze_changed_label_controls.py --source MTG-Jamendo \
  --manifest reproduced/mtg_jamendo_manifest.jsonl \
  --replay reproduced/mtg_jamendo_primary_prediction_replay.jsonl.gz \
  --report reports/mtg_jamendo_metadata_calibration.json \
  --output reproduced/mtg_changed_label_controls.json
```

The independent target uses the creators' clean consensus genre annotations on overlapping MTG tracks. No new human evaluation is collected for this paper. Download the pinned 4.27 MB annotation file and check its digest:

```sh
curl -L https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/cafd8e20c265ed84f1e61f1c875327971f43a62f/derived/music-classification-annotations/music-classification-annotations-clean.tsv -o data/mtg-jamendo/music-classification-annotations-clean.tsv
printf '%s  %s
' 9b060b3db785d475870ceb0682241782bf20053d751863fbd42e02cf11088e1d data/mtg-jamendo/music-classification-annotations-clean.tsv | shasum -a 256 -c -
PYTHONPATH=src python scripts/evaluate_independent_genre_target.py \
  --config configs/mtg_independent_genre_target.json \
  --annotations data/mtg-jamendo/music-classification-annotations-clean.tsv \
  --manifest reproduced/mtg_jamendo_manifest.jsonl \
  --replay reproduced/mtg_jamendo_primary_prediction_replay.jsonl.gz \
  --report reports/mtg_jamendo_metadata_calibration.json \
  --output reproduced/mtg_independent_genre_target.json
```

The frozen primary configuration contains three taxonomies. A post-audit check includes the fourth clean genre taxonomy with the same command and `--config configs/mtg_independent_genre_target_all_taxonomies_sensitivity.json --output reproduced/mtg_independent_genre_target_all_taxonomies_sensitivity.json`. Compare both fold contrasts and support counts with the packaged reports. The genre target is separate from the scorer's training labels but uses previously analyzed MTG tracks; the five-positive-block check is also post-audit.

`reports/target_masked_manuscript_summary.json` records all ten fold-level contrasts used in the manuscript table, including the matched-deletion checks. The reported Student-t intervals summarize the ten fixed held-out-group differences; training sets overlap.

## Check the additional genre results

The Song Describer check uses the official human-caption CSV at
`https://zenodo.org/records/10072001`; verify SHA-256
`fb853b0327394cfdbdc205f712fdb64df70e6fcd3bae3a9d3f1617d54b45ad70`.
The source is not bundled. The pinned config, evaluator, report, and score replay
are included. Recompute after generating the MTG manifest above:

```sh
PYTHONPATH=src:scripts python scripts/evaluate_song_describer_reference_check.py   --config configs/song_describer_reference_check_v1.json   --captions data/song_describer.csv   --mtg-manifest reproduced/mtg_jamendo_manifest.jsonl   --study-manifest reproduced/song_describer_manifest.jsonl   --replay reproduced/song_describer_recomputed.jsonl.gz   --output reproduced/song_describer_reference_check.json
```

The audio-derived feature study uses the official MTG-Jamendo
`raw_30s/acousticbrainz` release. Its 100 archive checksums and 55,699 track
checksums are bundled under `data/mtg_audio/`; downloaded shards are about
1.3 GB. Download them to a storage location with adequate space, then use
`scripts/extract_mtg_acoustic_features.py` to verify every archive and track
and produce the 80-coordinate feature file. The pinned evaluator uses
`numpy==2.4.6` and `scikit-learn==1.9.0`; no model selection uses consensus
labels. The packaged ten fold reports, prediction replays, and aggregate let
readers validate the scored result without downloading the large feature
release:

```sh
PYTHONPATH=src:scripts python scripts/aggregate_mtg_audio_feature_consequence.py   --config configs/mtg_audio_feature_consequence_v2.json   --fold-dir reports/mtg_audio_feature_consequence_v2/folds   --output reproduced/mtg_audio_feature_consequence_aggregate.json
cmp reports/mtg_audio_feature_consequence_v2/aggregate.json     reproduced/mtg_audio_feature_consequence_aggregate.json
```

Portable paths and the sanitized metadata report change the packaged audio
config and fold hashes from the private run. The archive rebinding preserves
all fold scores, condition targets, feature and prediction hashes, and the
descriptive interval values. The audio comparison uses precomputed statistics
from 30-second excerpts on the previously studied MTG tracks; it does not
measure full-song quality or listener preference.

## Reproduce the Free Music Archive catalog check

The official Free Music Archive metadata ZIP is about 342 MiB and is not
bundled. Download it from the URL pinned in
`configs/fma_reference_check_v1.json` and verify SHA-1
`f0df49ffe5f2a6008d7dc83c6915b31835dfe733`. The primary protocol was
fixed before inspecting this source. The included report and compressed
prediction replay permit independent checks without the download. The source
contains catalog text and genre tags, not listener or generated-audio outcomes.

```sh
mkdir -p data/fma reproduced/fma_reference_check_v1
curl -L https://os.unil.cloud.switch.ch/fma/fma_metadata.zip -o data/fma/fma_metadata.zip
printf '%s  %s
' f0df49ffe5f2a6008d7dc83c6915b31835dfe733 data/fma/fma_metadata.zip | shasum -a 1 -c -
PYTHONPATH=src python scripts/evaluate_fma_reference_check.py   --config configs/fma_reference_check_v1.json   --archive data/fma/fma_metadata.zip   --manifest reproduced/fma_reference_check_v1/manifest.jsonl   --replay reproduced/fma_reference_check_v1/predictions.jsonl.gz   --output reproduced/fma_reference_check_v1/report.json
```

The report gives all ten artist-fold decisions, score-tie-aware references,
source and replay hashes. The separate overlap analysis removed 90 FMA records
whose normalized title, artist, and album strings matched earlier MTG records
from fixed predictions. It is a post-audit sensitivity. After reproducing the
MTG manifest above, run:

```sh
PYTHONPATH=src python scripts/audit_fma_overlap_sensitivity.py   --fma-manifest reproduced/fma_reference_check_v1/manifest.jsonl   --mtg-manifest reproduced/mtg_jamendo_manifest.jsonl   --replay reproduced/fma_reference_check_v1/predictions.jsonl.gz   --output reproduced/fma_reference_check_v1/overlap_sensitivity.json
```

Compare the six decision intervals with the packaged primary and sensitivity
reports. Paths in the packaged reports may differ from local paths.
