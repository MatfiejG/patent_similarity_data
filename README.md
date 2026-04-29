# Patent Similarity Data and Measures

US patent similarity tools for training Doc2Vec vectors and exporting results in plain files instead of SQLite.

## Current workflow

`patent_d2v.py` now trains directly from a CSV test file and writes outputs to the `Outputs/` folder:

- `Outputs/patent_doc2v_12e.model`
- `Outputs/patent_doc2v_vectors.csv`
- `Outputs/patent_doc2v_vectors.jsonl`
- `Outputs/patent_doc2v_summary.json`

The script is currently configured to read:

- `Data/Test/all_patent_names_abstracts.csv`

The expected input columns are:

- `patent_id`
- `abstract` or `text`

## Install dependencies

Before running the scripts, install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

## Legacy data conversion

`write_sim_data_to_db.py` keeps its old filename for compatibility, but it no longer writes anything to SQLite.

It now converts legacy similarity files into notebook-friendly exports under `Outputs/legacy_exports/`:

- `cite_sims.csv` -> `cite_similarity.csv`
- `vectors.json` -> `vectors.csv` and `vectors.jsonl`
- `most_sim.json` -> `most_similar_pairs.csv` and `most_similar.jsonl`

## Loading the new vector export in a notebook

Example using the CSV file:

```python
import json
import pandas as pd

df = pd.read_csv("Outputs/patent_doc2v_vectors.csv")
df["vector"] = df["vector_json"].apply(json.loads)
df.head()
```

Example using the JSONL file:

```python
import pandas as pd

df = pd.read_json("Outputs/patent_doc2v_vectors.jsonl", lines=True)
df.head()
```

## Notes

- The current test CSV in this workspace is `all_patent_names_abstracts.csv`. A file named `all_patent_names.csv` was not present.
- The minimum token threshold was reduced so the abstract-only test data can be used for training.
- The original notebook in this repository still contains SQLite-based analysis examples and should be treated as legacy until those analysis steps are rewritten around file-based inputs.

## Reference

To reference this work, please cite: Whalen, R., Lungeanu, A., DeChurch, L. A., & Contractor, N. (2020). "Patent Similarity Data and Innovation Metrics." *Journal of Empirical Legal Studies.*
