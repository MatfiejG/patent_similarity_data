#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Legacy helper name retained for compatibility.

This script no longer writes similarity data into SQLite. Instead it converts
the project files into notebook-friendly CSV and JSONL exports.
"""

import csv
import json
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

OUTPUT_DIR = BASE_DIR / 'Outputs' / 'legacy_exports'


def export_citation_similarity():
    source_path = BASE_DIR / 'cite_sims.csv'
    target_path = OUTPUT_DIR / 'cite_similarity.csv'

    if not source_path.exists():
        print('Skipping citation similarities. File not found: ' + str(source_path))
        return False

    print('Exporting citation similarities to ' + str(target_path))

    with open(source_path, 'r', encoding='utf-8', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader, None)

        with open(target_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['patent_id', 'citation_id', 'similarity'])

            for row in reader:
                if len(row) < 3:
                    continue
                writer.writerow(row[:3])

    return True


def export_vectors():
    source_path = BASE_DIR / 'vectors.json'
    csv_path = OUTPUT_DIR / 'vectors.csv'
    jsonl_path = OUTPUT_DIR / 'vectors.jsonl'

    if not source_path.exists():
        print('Skipping vectors export. File not found: ' + str(source_path))
        return False

    print('Exporting vectors to ' + str(csv_path) + ' and ' + str(jsonl_path))

    with open(source_path, 'r', encoding='utf-8') as infile:
        with open(csv_path, 'w', encoding='utf-8', newline='') as csv_outfile:
            writer = csv.DictWriter(
                csv_outfile,
                fieldnames=['patent_id', 'vector_json'],
            )
            writer.writeheader()

            with open(jsonl_path, 'w', encoding='utf-8') as jsonl_outfile:
                for line in infile:
                    patent_id, vector = json.loads(line)

                    writer.writerow(
                        {
                            'patent_id': patent_id,
                            'vector_json': json.dumps(vector),
                        }
                    )
                    jsonl_outfile.write(
                        json.dumps(
                            {
                                'patent_id': patent_id,
                                'vector': vector,
                            }
                        )
                        + '\n'
                    )

    return True


def export_most_similar():
    source_path = BASE_DIR / 'most_sim.json'
    flat_csv_path = OUTPUT_DIR / 'most_similar_pairs.csv'
    jsonl_path = OUTPUT_DIR / 'most_similar.jsonl'

    if not source_path.exists():
        print('Skipping most-similar export. File not found: ' + str(source_path))
        return False

    print('Exporting most-similar data to ' + str(flat_csv_path) + ' and ' + str(jsonl_path))

    with open(source_path, 'r', encoding='utf-8') as infile:
        with open(flat_csv_path, 'w', encoding='utf-8', newline='') as csv_outfile:
            writer = csv.DictWriter(
                csv_outfile,
                fieldnames=['patent_id', 'rank', 'similar_patent_id', 'similarity'],
            )
            writer.writeheader()

            with open(jsonl_path, 'w', encoding='utf-8') as jsonl_outfile:
                for line in infile:
                    patent_id, similar_patents = json.loads(line)

                    jsonl_outfile.write(
                        json.dumps(
                            {
                                'patent_id': patent_id,
                                'top_similar': similar_patents,
                            }
                        )
                        + '\n'
                    )

                    for rank, pair in enumerate(similar_patents, start=1):
                        if len(pair) < 2:
                            continue

                        writer.writerow(
                            {
                                'patent_id': patent_id,
                                'rank': rank,
                                'similar_patent_id': pair[0],
                                'similarity': pair[1],
                            }
                        )

    return True


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        'citation_similarity': export_citation_similarity(),
        'vectors': export_vectors(),
        'most_similar': export_most_similar(),
    }

    print('Export complete: ' + json.dumps(results, indent=2))
