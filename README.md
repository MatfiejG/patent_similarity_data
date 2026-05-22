# Patent Similarity Data and Measures

US patent similarity tools for training Doc2Vec vectors and exporting results in plain files instead of SQLite.

## Current workflow

`patent_algo_d2v.py` trains directly from raw patent text CSV files and writes outputs to the `Outputs/` folder:

- `Outputs/patent_doc2v_12e.model`
- `Outputs/patent_doc2v_vectors.csv`
- `Outputs/patent_doc2v_vectors.jsonl`
- `Outputs/patent_doc2v_summary.json`

The script is currently configured to read and join:

- `Data/Raw/all_patent_names_description.csv`
- `Data/Raw/all_patent_names_claims.csv`

The expected input columns are:

- patent identifier: `Patent_number` or `patent_id`
- description/body text: `Description`, `description`, `body`, or `text`
- claims text: `claims`, `Claims`, `claim`, or `text`

For each patent number, the script combines the description first and the claims second into a single document before tokenization and Doc2Vec training.

During long CSV passes, `patent_algo_d2v.py` displays `tqdm` progress bars for:

- reading patent records for Doc2Vec training
- exporting trained patent vectors to CSV and JSONL

## Install dependencies

Before running the scripts, install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

The requirements include `tqdm`, which is used only for terminal progress bars.

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

## Vector-only similarity notebook

`patent_sim_data_vectors_only_notebook.ipynb` demonstrates the downstream analysis workflow using only the exported vector file:

- loads `Outputs/patent_doc2v_vectors.csv`
- validates that `patent_id` and `vector_json` are present
- parses each JSON-encoded vector into a NumPy array
- builds an in-memory patent ID to vector lookup
- provides `patent_pair_sim(patent1, patent2)` for cosine similarity between two selected patents
- computes all unique pairwise cosine similarities with `sklearn.metrics.pairwise.cosine_similarity`
- writes the pairwise results to `Outputs/patent_pairwise_similarity.parquet`, or to `Outputs/patent_pairwise_similarity.csv.gz` if Parquet support is unavailable
- plots and saves the similarity distribution as `Outputs/patent_similarity_distribution.png`

Because the notebook constructs the full cosine similarity matrix, memory use scales quadratically with the number of patents. It is suitable for exploratory work and moderate test exports; large patent collections may require batching, approximate nearest-neighbor search, or pairwise computation over selected subsets.

## Notes

- The current training corpus uses full raw description and claims files rather than the earlier abstract-only test file.
- The summary file records both input CSV paths and the configured identifier/text column names.
- The original notebook in this repository still contains SQLite-based analysis examples and should be treated as legacy until those analysis steps are rewritten around file-based inputs. The vectors-only notebook is the current file-based example for similarity analysis.

## Reference

To reference this work, please cite: Whalen, R., Lungeanu, A., DeChurch, L. A., & Contractor, N. (2020). "Patent Similarity Data and Innovation Metrics." *Journal of Empirical Legal Studies.*
