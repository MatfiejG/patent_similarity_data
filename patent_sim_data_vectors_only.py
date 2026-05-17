# Patent Vector Similarity Analysis (Vectors Only)
# This script loads the exported patent Doc2Vec vectors from `Outputs/patent_doc2v_vectors.csv`
# and computes cosine similarity directly from the loaded vectors.
# It does not require SQLite or external patent metadata.

import json
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame
import seaborn as sns
import pyarrow.parquet as pq
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Load the vectors
vectors_df = pd.read_csv('Outputs/patent_doc2v_vectors.csv')
meta_data: DataFrame = pd.read_csv('Data/Test//Dane ze starego papera.csv', delimiter=';')
patent_id_list = vectors_df['patent_id'].values
vectors = np.array([json.loads(v) for v in tqdm(vectors_df['vector_json'], desc='Loading vectors')]).astype(np.float32)

vector_map = dict(zip(patent_id_list, vectors))

patent_metadata_map = (
    meta_data.drop_duplicates('Number')
    .assign(Number=lambda df: df['Number'].astype(str))
    .set_index('Number')[['Group', 'Thicket ID', 'Thicket z csv']]
)

patent_id_array = pd.Series(patent_id_list).astype(str).values

patent_metadata_list = patent_metadata_map.reindex(patent_id_array)

patent_group_list = patent_metadata_list['Group'].values
patent_thicket_id_list = patent_metadata_list['Thicket ID'].values
patent_thicket_csv_list = patent_metadata_list['Thicket z csv'].values  

def patent_pair_sim(patent1, patent2):
    '''Return cosine similarity for two patent IDs from the loaded vectors.'''
    p1 = patent1
    p2 = patent2
    v1 = vector_map.get(p1)
    v2 = vector_map.get(p2)
    if v1 is None or v2 is None:
        return None
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return None
    return float(np.dot(v1, v2) / denom)

sample_ids = patent_id_list[:3]
print('Sample patent IDs:', sample_ids.tolist())
print('Sample similarity:', patent_pair_sim(sample_ids[0], sample_ids[1]))

n = len(vectors)
print(f'Computing all unique pairwise similarities for {n} patents. This will create roughly {n*(n-1)//2} pairs.')
with tqdm(desc='Computing similarity matrix') as pbar:
    sim_matrix = cosine_similarity(vectors, vectors, dense_output=True).astype(np.float32)
    pbar.update(1)

upper_i, upper_j = np.triu_indices(n, k=1)
pairwise_data = {
    'patent_id_1': patent_id_list[upper_i],
    'patent_id_2': patent_id_list[upper_j],
    'patent_group_1': patent_group_list[upper_i],
    'patent_group_2': patent_group_list[upper_j],
    'patent_thicket_id_1': patent_thicket_id_list[upper_i],
    'patent_thicket_id_2': patent_thicket_id_list[upper_j],
    'patent_thicket_csv_1': patent_thicket_csv_list[upper_i],
    'patent_thicket_csv_2': patent_thicket_csv_list[upper_j],
    'similarity': sim_matrix[upper_i, upper_j],
}
pairwise_df = pd.DataFrame(pairwise_data)

print('Created pairwise DataFrame with shape', pairwise_df.shape)
output_path = Path('Outputs/patent_pairwise_similarity.parquet')
try:
    print('Saving to parquet')
    pairwise_df.to_parquet(output_path, index=False)
    print(f'Saved pairwise results to {output_path}')

except Exception as e:
    print(f'Parquet save failed: {type(e).__name__}: {e}')
    output_path = Path('Outputs/patent_pairwise_similarity.csv.gz')
    print(f'Saving to CSV.GZ')
    pairwise_df.to_csv(output_path, index=False, compression='gzip')
    print(f'Saved pairwise results to {output_path}')

# fig, ax = plt.subplots(figsize=(10, 6))
# sns.histplot(pairwise_df['similarity'], bins=100, kde=True, stat='density', ax=ax, color='tab:blue')
# ax.set_title('Distribution of pairwise patent cosine similarities')
# ax.set_xlabel('Cosine similarity')
# ax.set_ylabel('Density')
# fig.tight_layout()
# fig.savefig('Outputs/patent_similarity_distribution.png', dpi=300)
# plt.show()
