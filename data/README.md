# Dataset Directory

This directory contains the local data used during model development
and evaluation.

Due to dataset licensing, privacy, and ethical restrictions, the
actual MRI image files are not included in this repository.

Expected data sources:

1. UTHealth-Endometriosis MRI Dataset
2. MOGaMBO cervical-cancer MRI dataset (proxy comparator, not healthy controls)
3. Collaborating Local Hospital Dataset

Place the datasets in the corresponding directories under `data/raw/`
when running the preprocessing pipeline.

The primary supervised experiment uses only T2 volumes from UT-EndoMRI and
MOGaMBO. MOGaMBO membership is a proxy-comparator label and does not establish
that a patient is clinically negative for endometriosis. The hospital cohort is
reserved for future external evaluation after its sequences and reference
labels are verified.

Preprocessing hashes decompressed voxel content before splitting. Exact duplicate
volumes are excluded and recorded in `data/processed/deduplication_report.csv`;
the effective sample count therefore refers to unique source volumes associated
with patient IDs, not independently re-identified unique people.
