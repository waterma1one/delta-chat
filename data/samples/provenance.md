# Sample provenance

The two source P&IDs used in development are different drawings, not revisions, and are not
redistributed here, so revision pairs are
synthesized with logged edits. Each edit is a gold change (`eval/datasets/delta_gold.jsonl`);
every authored token is round-trip validated as extractable by the native adapter.

| pair | split | edits |
|---|---|---|
| pair_01_native | train | tag 9038->9040; dim 150->200mm; delete note 2; renumber note 3->2; add 26-TIT-9211 |
| pair_02_hard | test | move 2nd of duplicate FV-9040 (disambiguation); split a note into two |
| pair_03_scanned | test | pair_01 rev B rasterized + skewed + noised + JPEG; native rev B = OCR truth |
| pair_04_multisheet | test | 3 sheets; tag edit on sheet 3 |
| pair_05_dxf | test | DXF: add LINE; change MTEXT tag; move block INSERT (gold bbox from ezdxf extents) |

Anti-leakage: TRAIN is used ONLY for threshold/note-gate calibration (see `select_calibration_pairs`);
TEST pairs headline the scorecard. Split + seed are read from `config/default.yaml`. No PII, no secrets.

Known limitation: the hard slice uses a duplicate-tag positional move (text), not a true vector-path
move; a genuine geometry-move synthesizer is future work. Q&A gold is a representative ~12-item set
(a fuller 50-200 set is future work).
