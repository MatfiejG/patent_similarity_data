#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import multiprocessing
import os
import re
import timeit
from pathlib import Path

import gensim
from gensim.models.doc2vec import TaggedDocument


start = timeit.default_timer()

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

INPUT_CSV = BASE_DIR / 'Data' / 'Test' / 'all_patent_names_abstracts.csv'
OUTPUT_DIR = BASE_DIR / 'Outputs'
MODEL_PATH = OUTPUT_DIR / 'patent_doc2v_12e.model'
VECTORS_CSV_PATH = OUTPUT_DIR / 'patent_doc2v_vectors.csv'
VECTORS_JSONL_PATH = OUTPUT_DIR / 'patent_doc2v_vectors.jsonl'
SUMMARY_PATH = OUTPUT_DIR / 'patent_doc2v_summary.json'

CSV_DELIMITER = ';'
TEXT_COLUMNS = ('abstract', 'text')
MIN_WORDS = 20
VECTOR_SIZE = 300
EPOCHS = 12


def tokenize_text(text):
    text = text.replace('\n', ' ').strip()
    words = re.findall(r"[\w']+|[.,!?;]", text)
    return [word.lower() for word in words]


def iter_patent_records(csv_path, text_columns, min_words, delimiter):
    seen_patents = set()

    with open(csv_path, 'r', encoding='utf-8', newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=delimiter)

        for row_number, row in enumerate(reader, start=1):
            if row_number % 100000 == 0:
                print('Processing patent ' + str(row_number))

            patent_id = (row.get('patent_id') or '').strip()
            if not patent_id:
                continue
            if patent_id.startswith('D'):
                continue
            if patent_id in seen_patents:
                continue

            text_value = ''
            for column in text_columns:
                candidate = row.get(column)
                if candidate:
                    text_value = candidate
                    break

            words = tokenize_text(text_value)
            if len(words) < min_words:
                continue

            seen_patents.add(patent_id)
            yield {
                'patent_id': patent_id,
                'token_count': len(words),
                'words': words,
            }


class DocIterator(object):
    """
    Streams corpus from a patent CSV export.
    """

    def __init__(self, csv_path, text_columns, min_words=20, delimiter=';'):
        self.csv_path = csv_path
        self.text_columns = text_columns
        self.min_words = min_words
        self.delimiter = delimiter

    def __iter__(self):
        for record in iter_patent_records(
            csv_path=self.csv_path,
            text_columns=self.text_columns,
            min_words=self.min_words,
            delimiter=self.delimiter,
        ):
            yield TaggedDocument(record['words'], [record['patent_id']])


def export_vectors(model, csv_path, text_columns, min_words, delimiter):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(VECTORS_CSV_PATH, 'w', encoding='utf-8', newline='') as csv_outfile:
        csv_writer = csv.DictWriter(
            csv_outfile,
            fieldnames=['patent_id', 'token_count', 'vector_json'],
        )
        csv_writer.writeheader()

        with open(VECTORS_JSONL_PATH, 'w', encoding='utf-8') as jsonl_outfile:
            exported = 0

            for record in iter_patent_records(
                csv_path=csv_path,
                text_columns=text_columns,
                min_words=min_words,
                delimiter=delimiter,
            ):
                patent_id = record['patent_id']
                vector = [float(value) for value in model.dv[patent_id]]
                vector_json = json.dumps(vector)

                csv_writer.writerow(
                    {
                        'patent_id': patent_id,
                        'token_count': record['token_count'],
                        'vector_json': vector_json,
                    }
                )
                jsonl_outfile.write(
                    json.dumps(
                        {
                            'patent_id': patent_id,
                            'token_count': record['token_count'],
                            'vector': vector,
                        }
                    )
                    + '\n'
                )
                exported += 1

    summary = {
        'input_csv': str(csv_path),
        'vector_count': exported,
        'vector_size': model.vector_size,
        'epochs': model.epochs,
        'min_words': min_words,
        'text_columns': list(text_columns),
        'csv_delimiter': delimiter,
        'model_path': str(MODEL_PATH),
        'vectors_csv_path': str(VECTORS_CSV_PATH),
        'vectors_jsonl_path': str(VECTORS_JSONL_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    if not INPUT_CSV.exists():
        raise FileNotFoundError('Input CSV not found: ' + str(INPUT_CSV))

    workers = multiprocessing.cpu_count()
    doc_iterator = DocIterator(
        csv_path=INPUT_CSV,
        text_columns=TEXT_COLUMNS,
        min_words=MIN_WORDS,
        delimiter=CSV_DELIMITER,
    )

    model = gensim.models.Doc2Vec(
        documents=doc_iterator,
        workers=workers,
        vector_size=VECTOR_SIZE,
        epochs=EPOCHS,
    )

    if hasattr(model, 'delete_temporary_training_data'):
        model.delete_temporary_training_data(
            keep_doctags_vectors=True,
            keep_inference=True,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_PATH))

    summary = export_vectors(
        model=model,
        csv_path=INPUT_CSV,
        text_columns=TEXT_COLUMNS,
        min_words=MIN_WORDS,
        delimiter=CSV_DELIMITER,
    )

    print('Saved model to ' + str(MODEL_PATH))
    print('Saved vectors to ' + str(VECTORS_CSV_PATH))
    print('Saved JSONL vectors to ' + str(VECTORS_JSONL_PATH))
    print('Summary: ' + json.dumps(summary, indent=2))


stop = timeit.default_timer()
total_time = stop - start
mins, secs = divmod(total_time, 60)
hours, mins = divmod(mins, 60)

print("Total running time: %d:%d:%d. \n" % (hours, mins, secs))
