# Researcher Notes

This project was changed so you can test the patent vector workflow without needing a SQLite database.

## What changed

`patent_algo_d2v.py` used to read patent text from SQL tables and, later, from an abstract-only test CSV. It now reads directly from the full raw CSV files:

- `Data/Raw/all_patent_names_description.csv`
- `Data/Raw/all_patent_names_claims.csv`

For each `Patent_number`, the script joins the description first and the claims second into one document before Doc2Vec training.

The script used to save vectors back into SQLite. It now saves them as regular files in the `Outputs` folder:

- `patent_doc2v_vectors.csv`
- `patent_doc2v_vectors.jsonl`
- `patent_doc2v_summary.json`
- `patent_doc2v_12e.model`

## Why this helps

You can now inspect the outputs without any database software.

- The CSV file is easy to open in spreadsheet tools, pandas, or text editors.
- The JSONL file is convenient for notebooks and for loading one patent vector per line.
- The summary file records which input file and settings were used.

## Important practical difference

The workflow has moved from the abstract-only test data to fuller patent text. The resulting vectors should therefore reflect substantially more technical content than the earlier test run, because each patent document includes both the description/body text and the claims when both are available.

## Legacy helper script

`write_sim_data_to_db.py` keeps its old name, but it no longer writes to SQLite.

Instead, it converts older similarity files into easier-to-read CSV and JSONL exports in:

- `Outputs/legacy_exports/`

## What to do next

If you want to continue working only with CSV files, the next natural step is to rewrite the notebook analysis so it loads:

- `Outputs/patent_doc2v_vectors.csv` or `Outputs/patent_doc2v_vectors.jsonl`

instead of reading the `doc2vec` table from SQLite.

For large full-text runs, monitor runtime and memory use. The script streams descriptions during training/export and loads the claims file into a patent-number lookup so each description can be combined with its matching claims.
