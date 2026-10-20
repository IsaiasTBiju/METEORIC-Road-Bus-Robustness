# Data

This folder documents the datasets used by METEORIC.

## Planned local structure

- `raw/` — original downloaded datasets, kept unchanged
- `interim/` — temporary files created during cleaning and matching
- `processed/` — validated files prepared for analysis

These folders will be created as needed during the project.

## Handling rules

- Keep original datasets unchanged.
- Record each dataset’s source, download date, licence, and relevant limitations.
- Do not upload raw datasets to GitHub until their reuse terms have been checked.
- Do not commit credentials, personal data, or files that the data provider does not permit sharing.
- Keep scripts and documentation that explain how permitted derived data was produced.