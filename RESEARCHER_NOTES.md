# Researcher Notes

This project was changed so you can test the patent vector workflow without needing a SQLite database.

## What changed

`patent_d2v.py` used to read patent text from SQL tables. It now reads directly from the CSV file in `Data/Test/all_patent_names_abstracts.csv`.

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

The available test file contains abstracts, not full patent descriptions plus claims. Because abstracts are much shorter, the script now accepts shorter texts during training.

This means the vectors are useful for testing the workflow, but they are not exactly the same as vectors trained on the original full-text patent database.

## Legacy helper script

`write_sim_data_to_db.py` keeps its old name, but it no longer writes to SQLite.

Instead, it converts older similarity files into easier-to-read CSV and JSONL exports in:

- `Outputs/legacy_exports/`

## What to do next

If you want to continue working only with CSV files, the next natural step is to rewrite the notebook analysis so it loads:

- `Outputs/patent_doc2v_vectors.csv` or `Outputs/patent_doc2v_vectors.jsonl`

instead of reading the `doc2vec` table from SQLite.

If you later want research-grade vectors rather than a test run, you should prepare a richer input CSV with longer patent text, such as abstracts plus claims or full descriptions.
